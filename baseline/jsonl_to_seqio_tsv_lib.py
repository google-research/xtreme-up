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
"""Utility functions for converting standard JSONL files into model input TSVs.

The JSONL files are intended to be free of any model-specific assumptions.

The TSV files may contain model-specific or system-specific information such as
instructions, prompts, tokenization, or other formatting.
"""
import dataclasses
import json
import os
import re

from typing import Any, Callable, Iterable

from xtreme_up.evaluation import file_utils


def process_example(
    example_dict: dict[str, Any],
    output_column_order: Iterable[str],
    debug_path: str,
) -> str:
  """Turns a JSON dict for a single example into a TSV str for that example."""
  result_cols = []
  for column in output_column_order:
    if column not in example_dict:
      raise ValueError(
          f'Column "{column}" not found in file "{debug_path}": '
          + str(example_dict)
      )
    result_cols.append(example_dict[column])
  return '\t'.join(result_cols)


@dataclasses.dataclass(frozen=True)
class TaskInfo:
  output_column_order: Iterable[str] = ('input', 'target')
  custom_processor: None | Callable[[dict[str, Any], str], dict[str, str]] = (
      None
  )


def preprocess_autocomplete_example(
    json_dict: dict[str, Any], filename: str
) -> dict[str, str]:
  """Adds language and scripts prefix to the `inputs` features."""

  assert '.jsonl' in filename
  language = filename.replace('.jsonl', '')

  json_dict['context'] = language + '-' + json_dict['context']

  # Append special character '$' to help tokenizer recognize partially completed
  # words (otherwise the distinction between space and no space following a word
  # gets flattened by SentencePiece's trimming normalization.
  json_dict['context'] = json_dict['context'] + '$'
  return json_dict


def get_task_info(task_name: str) -> TaskInfo:
  if 'qa'in task_name:
    task_name = 'qa'
  if task_name == 'retrieval_index':
    return NON_DEFAULT_TASK_INFO[task_name]
  if 'retrieval'in task_name:
    task_name = 'retrieval'
  if task_name in NON_DEFAULT_TASK_INFO:
    return NON_DEFAULT_TASK_INFO[task_name]
  else:
    return TaskInfo()


NER_DELIMITER = '$$'


def byte_slice(text: str, start_byte: int, limit_byte: int) -> str:
  return text.encode()[start_byte:limit_byte].decode()


def preprocess_ner_example(
    json_dict: dict[str, Any], filename: str
) -> dict[str, str]:
  """Performs preprocessing for the NER task."""
  del filename  # Unused.

  text = json_dict['text']
  spans = []
  for span in json_dict['spans']:
    entity = byte_slice(text, span['start_byte'], span['limit_byte'])
    label = span['label']
    spans.append(f'{label}: {entity}')
  target = f' {NER_DELIMITER} '.join(spans)
  # Normalize tabs in input data.
  text = text.replace('\t', ' ')
  target = target.replace('\t', ' ')
  return {
      'input': text,
      'target': target,
  }


def preprocess_qa_example(
    json_dict: dict[str, str], filename: str
) -> dict[str, str]:
  """Performs preprocessing for the QA task."""
  del filename  # Unused.

  question = json_dict['question']
  title = json_dict['title']
  context = json_dict['context']
  target = json_dict['target']
  return {
      'input': f'question: {question} title: {title} context: {context}',
      'target': target,
  }


def preprocess_retrieval_example(
    json_dict: dict[str, Any], filename: str
) -> dict[str, str]:
  """Performs preprocessing for the retrieval task."""
  del filename  # Unused.

  return {k: str(v) for k, v in json_dict.items()}


def preprocess_transliteration_example(
    json_dict: dict[str, Any], filename: str
) -> dict[str, str]:
  assert 'jsonl' in filename
  match = re.search(r'([A-Za-z]+)2([A-Za-z]+)\.([a-z]+)\.jsonl', filename)
  if not match:
    raise ValueError(f'Unrecognized file pattern: {filename}')
  src_script, tgt_script, language = match.groups()
  return {
      'input': f'{language}-{src_script}-{tgt_script} ' + json_dict['input'],
      'target': json_dict['target'],
  }


NON_DEFAULT_TASK_INFO: dict[str, TaskInfo] = {
    'qa': TaskInfo(
        output_column_order=['input', 'target'],
        custom_processor=preprocess_qa_example,
    ),
    'retrieval': TaskInfo(
        output_column_order=['id', 'title', 'context', 'question'],
        custom_processor=preprocess_retrieval_example,
    ),
    'retrieval_index': TaskInfo(
        # Index doesn't have questions since it's just a retrieval pool.
        output_column_order=['id', 'title', 'context'],
        custom_processor=preprocess_retrieval_example,
    ),
    'autocomplete': TaskInfo(
        output_column_order=['context', 'target'],
        custom_processor=preprocess_autocomplete_example,
    ),
    'ner': TaskInfo(
        output_column_order=['input', 'target'],
        custom_processor=preprocess_ner_example,
    ),
    'transliteration': TaskInfo(
        output_column_order=['input', 'target'],
        custom_processor=preprocess_transliteration_example,
    ),
}


def get_tsv_lines(path: str, task_info: TaskInfo) -> list[str]:
  """Generates preprocessed TSV lines for a particular task."""
  tsv_lines = []
  with file_utils.open(path, 'r') as f:
    for i, line in enumerate(f):
      if not line.strip():
        continue
      try:
        example = json.loads(line)
      except json.JSONDecodeError as e:
        raise ValueError(f"Bad JSON line at {path}:{i+1}: '{line}'") from e
      if task_info.custom_processor:
        filename = os.path.basename(path)
        example = task_info.custom_processor(example, filename)
      tsv_line = process_example(
          example, task_info.output_column_order, debug_path=path
      )
      tsv_lines.append(tsv_line)
  return tsv_lines
