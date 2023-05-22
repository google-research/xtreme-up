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
"""Utility methods for XTREME-UP tasks."""
import os
from typing import Mapping

import gin
from multilingual_t5 import vocab as mt5_vocab
import t5.data


# Import gin/tasks_lib.gin and pass on the command line as:
# --gin.BENCHMARK_TSV_DATA_DIR='x'
@gin.configurable
def benchmark_tsv_data_dir(value: str = '') -> str:
  """Input dir with preprocessed task/split/*.tsv from jsonl_to_tsv.py."""
  return value


# Import gin/tasks_lib.gin and pass on the command line as:
# --gin.TASK_TSV_DATA_DIR='x'
@gin.configurable
def task_tsv_data_dir(value: str = '') -> str:
  """Override input dir with preprocessed split/*.tsv from jsonl_to_tsv.py."""
  return value


DEFAULT_SPM_PATH = mt5_vocab.DEFAULT_SPM_PATH
DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)
DEFAULT_OUTPUT_FEATURES = {
    'inputs': t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False
    ),
    'targets': t5.data.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True),
}

BYT5_OUTPUT_FEATURES = {
        'inputs': t5.data.Feature(vocabulary=t5.data.ByteVocabulary()),
        'targets': t5.data.Feature(vocabulary=t5.data.ByteVocabulary()),
    }

CANONICAL_SPLITS = ['train', 'validation', 'test']


def get_output_features(model: str) -> Mapping[str, t5.data.Feature]:
  if model.lower() == 'byt5':
    return BYT5_OUTPUT_FEATURES
  if model.lower() in ('mt5', 'nmt5'):
    return DEFAULT_OUTPUT_FEATURES
  raise ValueError(f'{model} is not supported.')


def get_data_dir(task: str) -> str:
  if task_tsv_data_dir():
    return task_tsv_data_dir()
  else:
    return os.path.join(benchmark_tsv_data_dir(), task)


def get_files_by_split(task: str, lang: str) -> dict[str, str]:
  task_data_dir = get_data_dir(task)
  return {
      split: os.path.join(task_data_dir, split, f'{lang}.tsv')
      for split in CANONICAL_SPLITS
  }


def get_index_split(task: str) -> dict[str, str]:
  task_data_dir = get_data_dir(task)
  return {'index': os.path.join(task_data_dir, 'index.tsv')}
