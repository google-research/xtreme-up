# coding=utf-8
# Copyright 2023 The Google Research authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Writes nearest neighbors for each language for the XTREME-UP retrieval task."""

from collections.abc import Sequence
import json
from typing import Optional

from absl import app
from absl import flags
import jax  # jax is used only for accelerated numpy and not for any modeling.
import jax.numpy as jnp

from xtreme_up.evaluation import constants
from xtreme_up.evaluation import file_utils


_QUERY_DIR = flags.DEFINE_string(
    'query_dir', None, 'Path to jsonl file of query embeddings.'
)

_CANDIDATE_FILE = flags.DEFINE_string(
    'candidate_file', None, 'Path to jsonl file of candidate embeddings.'
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Path to directory to write per-language results.'
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    None,
    'Name of the model, to be used in determining input filesnames to read.',
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    (
        'Task we are evaluation, either "retrieval_in_lang" or'
        ' "retrieval_cross_lang".'
    ),
)

_L2_NORMALIZE = flags.DEFINE_bool(
    'l2_normalize',
    True,
    'Whether to normalize embeddings before computing dot product similarity.',
)


def read_embedding_file(
    data_f: str, lang: Optional[str] = None
) -> tuple[list[str], jnp.ndarray]:
  """Reads inference files and returns ids and embeddings."""
  keys = []
  embs = []
  with file_utils.open(data_f, 'rb') as fin:
    for l in fin:
      ex = json.loads(l)
      if lang:
        key = lang + '-' + ex['inputs']['targets_pretokenized']
      else:
        key = ex['inputs']['targets_pretokenized']
      emb = ex['score']  # list of floats
      keys.append(key)
      embs.append(emb)
  embs = jnp.array(embs)
  return keys, embs


def l2_normalize(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
  """L2 norm of the input."""
  x = x / jnp.clip(
      jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True), a_min=1e-9
  )
  return x


def compute_similarities(
    queries: jnp.ndarray, candidates: jnp.ndarray, normalize: bool = True
) -> jnp.ndarray:
  """Compute nearest neighbors."""

  if normalize:  # simiarlity is cosine
    queries = l2_normalize(queries)
    candidates = l2_normalize(candidates)

  scores = jnp.matmul(queries, jnp.transpose(candidates, axes=(1, 0)))
  _, neighbors = jax.lax.top_k(scores, 10)
  return neighbors


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Find nearest neighbors.
  langs = constants.get_languages(_TASK.value)

  query_keys = []
  queries = []
  for lang in langs:
    query_key, query = read_embedding_file(
        _QUERY_DIR.value
        + f'/xtreme_up_{_TASK.value}_inference_query_{lang}_{_MODEL_NAME.value}-score.jsonl-00000-of-00001',
        lang
    )
    query_keys.extend(query_key)
    queries.append(query)
  queries = jnp.vstack(queries)
  candidate_keys, candidates = read_embedding_file(_CANDIDATE_FILE.value)
  neighbors = compute_similarities(queries, candidates, _L2_NORMALIZE.value)

  output_lines = {}
  for lang in langs:
    output_lines[lang] = []

  for query_key, query_neighbors_idx in zip(query_keys, neighbors):
    lang = query_key.split('-')[0]
    query_neighbors = [candidate_keys[idx] for idx in query_neighbors_idx]
    # query_key = gold key; query_neighbors = predictions
    json_line = json.dumps(
        {'target': query_key[3:], 'prediction': query_neighbors}
    )
    output_lines[lang].append(json_line)

  for lang in langs:
    f = file_utils.open(_OUTPUT_DIR.value + '/nn_' + lang + '.jsonl', 'w')
    json_lines = output_lines[lang]
    for json_line in json_lines:
      f.write(json_line + '\n')
    f.close()


if __name__ == '__main__':
  app.run(main)
