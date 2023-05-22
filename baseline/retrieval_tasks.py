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
"""XTREME-UP retrieval tasks."""
import functools
from typing import List, Mapping

import seqio
import tensorflow as tf
from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants


def split_and_rekey_processor(dataset: tf.data.Dataset,
                              has_query: bool = True) -> tf.data.Dataset:
  """Prepare texts for retrieval tasks."""
  split_map_fn = lambda x: tf.strings.split(x, sep="\t", maxsplit=-1)

  def rekey_map_fn(x: List[str]) -> Mapping[str, str]:
    qid = x[0]
    title = x[1]
    candidate = x[2]

    if has_query:
      query = x[3]
    else:
      query = ""

    candidate = title + " " + candidate
    return {"id": qid, "query": query, "candidate": candidate}

  dataset = dataset.map(split_map_fn)
  return dataset.map(rekey_map_fn)


def to_inference_pair(dataset: tf.data.Dataset, split: str) -> tf.data.Dataset:
  """Transformas data to a single text (either query or candidate) and its id."""

  def _to_inference_pair(x: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    example = {"targets": x["id"]}
    if split == "query":
      example.update({"inputs": x["query"]})
    else:
      example.update({"inputs": x["candidate"]})
    return example

  return dataset.map(_to_inference_pair)


def lowercase(dataset: tf.data.Dataset,
              features: List[str],
              override_feature: bool = False) -> tf.data.Dataset:
  """Lowercase features values specified in |features|."""

  def _lowercase(example):
    for feature in features:
      key = feature if override_feature else f"{feature}_lower"
      example[key] = tf.strings.lower(example[feature])
    return example

  return dataset.map(_lowercase, num_parallel_calls=tf.data.AUTOTUNE)

for model in ('mt5', 'byt5'):
  # In-language tasks
  tydi_train_tasks = []
  for lang in constants.get_languages(task='retrieval_in_lang'):
    task_name = f'xtreme_up_retrieval_in_lang_train_{lang}_{model}'
    tydi_train_tasks.append(task_name)
    split_to_filepattern = tasks_lib.get_files_by_split(
        'retrieval_in_lang', lang
    )
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern),
        preprocessors=[
            functools.partial(
                split_and_rekey_processor, has_query=True
            ),
            functools.partial(
                seqio.preprocessors.rekey,
                key_map={'inputs': 'query', 'targets': 'candidate'},
            ),
            functools.partial(
                lowercase,
                features=['inputs', 'targets'],
                override_feature=True,
            ),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=tasks_lib.get_output_features(model),
    )
  seqio.MixtureRegistry.add(
      f'xtreme_up_retrieval_in_lang_train_{model}',
      tydi_train_tasks,
      default_rate=1.0,
  )

  # Perform inference to generate embeddings of passages candidates.
  in_lang_index_by_split = tasks_lib.get_index_split('retrieval_in_lang')
  seqio.TaskRegistry.add(
      f'xtreme_up_retrieval_in_lang_inference_candidate_{model}',
      source=seqio.TextLineDataSource(
          split_to_filepattern=in_lang_index_by_split
      ),
      preprocessors=[
          functools.partial(
              split_and_rekey_processor, has_query=False
          ),
          functools.partial(
              to_inference_pair, split='candidate'
          ),
          functools.partial(
              lowercase,
              features=['inputs'],
              override_feature=True,
          ),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=tasks_lib.get_output_features(model),
  )

  # Perform inference to generate embeddings of queries.
  tydi_query_tasks = []
  for lang in constants.get_languages(task='retrieval_in_lang'):
    task_name = f'xtreme_up_retrieval_in_lang_inference_query_{lang}_{model}'
    tydi_query_tasks.append(task_name)
    split_to_filepattern = tasks_lib.get_files_by_split(
        'retrieval_in_lang', lang
    )
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern),
        preprocessors=[
            functools.partial(
                split_and_rekey_processor, has_query=True
            ),
            functools.partial(
                to_inference_pair, split='query'
            ),
            functools.partial(
                lowercase,
                features=['inputs'],
                override_feature=True,
            ),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=tasks_lib.get_output_features(model),
    )
  seqio.MixtureRegistry.add(
      f'xtreme_up_retrieval_in_lang_inference_query_{model}',
      tydi_query_tasks,
      default_rate=1.0,
  )

  # Cross-language tasks
  xor_train_tasks = []
  for lang in constants.get_languages(task='retrieval_cross_lang'):
    task_name = f'xtreme_up_retrieval_cross_lang_train_{lang}_{model}'
    xor_train_tasks.append(task_name)
    split_to_filepattern = tasks_lib.get_files_by_split(
        'retrieval_cross_lang', lang
    )
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern),
        preprocessors=[
            functools.partial(
                split_and_rekey_processor, has_query=True
            ),
            functools.partial(
                seqio.preprocessors.rekey,
                key_map={'inputs': 'query', 'targets': 'candidate'},
            ),
            functools.partial(
                lowercase,
                features=['inputs', 'targets'],
                override_feature=True,
            ),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=tasks_lib.get_output_features(model),
    )
  seqio.MixtureRegistry.add(
      f'xtreme_up_retrieval_cross_lang_train_{model}',
      xor_train_tasks,
      default_rate=1.0,
  )

  # Perform inference to generate embeddings of passages candidates.
  cross_lang_index_by_split = tasks_lib.get_index_split('retrieval_cross_lang')
  seqio.TaskRegistry.add(
      f'xtreme_up_retrieval_cross_lang_inference_candidate_{model}',
      source=seqio.TextLineDataSource(
          split_to_filepattern=cross_lang_index_by_split
      ),
      preprocessors=[
          functools.partial(
              split_and_rekey_processor, has_query=False
          ),
          functools.partial(
              to_inference_pair, split='candidate'
          ),
          functools.partial(
              lowercase,
              features=['inputs'],
              override_feature=True,
          ),
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=tasks_lib.get_output_features(model),
  )

  # Perform inference to generate embeddings of queries.
  xor_query_tasks = []
  for lang in constants.get_languages(task='retrieval_cross_lang'):
    task_name = f'xtreme_up_retrieval_cross_lang_inference_query_{lang}_{model}'
    xor_query_tasks.append(task_name)
    split_to_filepattern = tasks_lib.get_files_by_split(
        'retrieval_in_lang', lang
    )
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern),
        preprocessors=[
            functools.partial(
                split_and_rekey_processor, has_query=True
            ),
            functools.partial(
                to_inference_pair, split='query'
            ),
            functools.partial(
                lowercase,
                features=['inputs'],
                override_feature=True,
            ),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=tasks_lib.get_output_features(model),
    )
  seqio.MixtureRegistry.add(
      f'xtreme_up_retrieval_cross_lang_inference_query_{model}',
      xor_query_tasks,
      default_rate=1.0,
  )
