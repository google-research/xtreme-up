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
r"""Script to convert released JSONL files into TSVs consumable simply by SeqIO.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags

from xtreme_up.baseline import jsonl_to_seqio_tsv_lib
from xtreme_up.evaluation import file_utils


TASKS = [
    'ner',
    'asr',
    'translation',
    'transliteration',
    'qa_in_lang',
    'qa_cross_lang',
    'retrieval_in_lang',
    'retrieval_cross_lang',
    'semantic_parsing',
    'autocomplete',
]


_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Input directory containing *.jsonl files.',
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Output directory to write *.tsv files to.',
    required=True,
)

_TASK = flags.DEFINE_enum(
    'task',
    None,
    TASKS,
    (
        'Task name; used to look up processing defined in '
        "jsonl_to_seqio_tsv.py. If your task isn't listed as a valid option, "
        "add it to this flag's enum and NON_DEFAULT_TASK_INFO if necessary."
    ),
    required=True,
)


SPLITS = ['train', 'validation', 'test']


def convert_dir(jsonl_in_dir: str, tsv_out_dir: str, task: str) -> None:
  """Converts a directory of JSONL files to TSV (all languages in a split."""
  if not file_utils.exists(tsv_out_dir):
    file_utils.makedirs(tsv_out_dir)

  input_jsonl_basenames = [
      f for f in file_utils.listdir(jsonl_in_dir) if f.endswith('.jsonl')
  ]
  output_tsv_basenames = [
      f[:f.rfind('jsonl')] + 'tsv' for f in input_jsonl_basenames
  ]
  existing_output_files = file_utils.listdir(tsv_out_dir)
  for output_basename in output_tsv_basenames:
    assert (
        output_basename not in existing_output_files
    ), f'{output_basename} already exists in {tsv_out_dir}'

  for input_basename, output_basename in zip(
      input_jsonl_basenames, output_tsv_basenames, strict=True
  ):
    input_path = os.path.join(jsonl_in_dir, input_basename)
    output_path = os.path.join(tsv_out_dir, output_basename)
    convert_file(input_path, output_path, task=task)


def convert_file(input_path: str, output_path: str, task: str) -> None:
  """Converts a single file (language) from JSONL to TSV."""
  task_info = jsonl_to_seqio_tsv_lib.get_task_info(task)
  tsv_lines = jsonl_to_seqio_tsv_lib.get_tsv_lines(input_path, task_info)
  with file_utils.open(output_path, 'w') as f:
    for line in tsv_lines:
      print(line, file=f)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  splits = list(SPLITS)
  for split in splits:
    convert_dir(
        jsonl_in_dir=os.path.join(_INPUT_DIR.value, split),
        tsv_out_dir=os.path.join(_OUTPUT_DIR.value, split),
        task=_TASK.value,
    )
  if 'retrieval' in _TASK.value:
    # For the retrieval task, also convert the retrieval index, which is a
    # single file for all languages and splits (but is separate per task).
    # The index consists of the positive and negative examples for the
    # validation and test splits.
    input_path = os.path.join(_INPUT_DIR.value, 'index.jsonl')
    output_path = os.path.join(_OUTPUT_DIR.value, 'index.tsv')
    convert_file(input_path, output_path, task='retrieval_index')


if __name__ == '__main__':
  app.run(main)
