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
"""Takes the system output and writes the prediction to the submission folder.

It is currently setup to take outputs with filenames formatted by T5X/seqio
and task names as setup in the XTREME-UP baseline and move them to standard
locations. If you use a different framework, this script is a good place to
normalize any differences in file patterns.
"""

from collections.abc import Sequence
import json
import os

from absl import app
from absl import flags
from absl import logging

from xtreme_up.evaluation import constants
from xtreme_up.evaluation import file_utils

_TASK = flags.DEFINE_string(
    'task',
    None,
    'The name of the task or a comma-delimited list.',
    required=True,
)

_SYSTEM_OUTPUT_DIR = flags.DEFINE_string(
    'system_output_dir',
    None,
    'Directory containing the output of inference or a comma-delimited list of '
    'such directories.',
    required=True,
)

_SUBMISSION_DIR = flags.DEFINE_string(
    'submission_dir', None,
    'The unique name for a submission. It is also the folder name containing '
    'all the outputs.',
    required=True,
)

_SPLIT = flags.DEFINE_enum(
    'split',
    None,
    ['validation', 'test'],
    'Which split is being evaluated.',
    required=True,
)

_CHECKPOINT_STEP = flags.DEFINE_string(
    'checkpoint_step',
    None,
    (
        'The checkpoint step used for generating the outputs, or a'
        ' comma-delimited list of such checkpoints.'
    ),
    required=True,
)

_MODEL = flags.DEFINE_enum(
    'model', None, ['mt5', 'byt5'],
    'The model name (if it is part of the task name).',
    required=True,
)


def system_out_to_submission(system_filename: str,
                             submission_filename: str) -> None:
  logging.info('`%s` -> `%s` ...', system_filename, submission_filename)
  parent = os.path.dirname(submission_filename)
  file_utils.makedirs(parent)
  with file_utils.open(submission_filename, 'w') as submission_f:
    with file_utils.open(system_filename, 'r') as system_f:
      for line in system_f:
        system_output = json.loads(line.strip())
        submission_data = {'prediction': system_output['prediction']}
        submission_f.write(json.dumps(submission_data) + '\n')


def move_transliteration_files(
    system_output_dir: str, submission_dir: str, split: str, checkpoint: int
) -> None:
  """Defines the system output filenames and moves them to the submission directory.
  """
  for lang, scripts in constants.TRANSLIT_LANGS_AND_SCRIPTS.items():
    for src_script, trg_script in scripts:
      system_filename = os.path.join(
          system_output_dir,
          f'xtreme_up_transliteration_{lang}_{src_script}_{trg_script}_{_MODEL.value}-{checkpoint}.jsonl',
      )
      submission_filename = os.path.join(
          submission_dir,
          'transliteration',
          split,
          f'{src_script}2{trg_script}.{lang}.jsonl',
      )
      system_out_to_submission(system_filename, submission_filename)
    for trg_script, src_script in scripts:
      system_filename = os.path.join(
          system_output_dir,
          f'xtreme_up_transliteration_{lang}_{src_script}_{trg_script}_{_MODEL.value}-{checkpoint}.jsonl',
      )
      submission_filename = os.path.join(
          submission_dir,
          'transliteration',
          split,
          f'{src_script}2{trg_script}.{lang}.jsonl',
      )
      system_out_to_submission(system_filename, submission_filename)


def move_mt_files(
    system_output_dir: str, submission_dir: str, split: str, checkpoint: int
) -> None:
  """Defines the system output filenames and move them to the submission directory.
  """
  for lang in constants.get_languages(
      task='translation', under_represented_only=True
  ):
    if _MODEL.value == 'mt5':
      system_filename = os.path.join(
          system_output_dir,
          f'xtreme_up_translation_en_{lang}-{checkpoint}.jsonl')
    else:
      system_filename = os.path.join(
          system_output_dir,
          f'xtreme_up_translation_en_{lang}_{_MODEL.value}-{checkpoint}.jsonl')
    submission_filename = os.path.join(
        submission_dir, 'translation', split, f'en2{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_qa_cross_language_files(
    system_output_dir: str, submission_dir: str, split: str, checkpoint: int
) -> None:
  """Defines the system output filenames and move them to the submission directory."""
  for lang in constants.get_languages(
      task='qa_cross_lang', under_represented_only=True
  ):
    system_filename = (
        system_output_dir
        + '/'
        + f'xtreme_up_qa_cross_lang.{lang}_{_MODEL.value}-{checkpoint}.jsonl'
    )
    submission_filename = os.path.join(
        submission_dir, 'qa_cross_lang', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_qa_in_language_files(
    system_output_dir: str, submission_dir: str, split: str, checkpoint: int
) -> None:
  """Defines the system output filenames and move them to the submission directory."""
  for lang in constants.get_languages(
      task='qa_in_lang', under_represented_only=True
  ):
    system_filename = (
        system_output_dir
        + '/'
        + f'xtreme_up_qa_in_lang.{lang}_{_MODEL.value}-{checkpoint}.jsonl'
    )
    submission_filename = os.path.join(
        submission_dir, 'qa_in_lang', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_retrieval_cross_language_files(
    system_output_dir: str, submission_dir: str, split: str
) -> None:
  """Defines the system output filenames and move them to the submission directory."""
  for lang in constants.get_languages(
      task='retrieval_cross_lang', under_represented_only=True
  ):
    system_filename = system_output_dir + '/' + f'nn_{lang}.jsonl'
    submission_filename = os.path.join(
        submission_dir, 'retrieval_cross_lang', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_retrieval_in_language_files(
    system_output_dir: str, submission_dir: str, split: str
) -> None:
  """Defines the system output filenames and move them to the submission directory."""
  for lang in constants.get_languages(
      task='retrieval_in_lang', under_represented_only=True
  ):
    system_filename = system_output_dir + '/' + f'nn_{lang}.jsonl'
    submission_filename = os.path.join(
        submission_dir, 'retrieval_in_lang', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_ner_files(
    system_output_dir: str,
    submission_dir: str,
    model: str,
    split: str,
    checkpoint: int,
) -> None:
  """Defines the system output filenames and move them to the submission directory.
  """
  for lang in constants.get_languages(task='ner', under_represented_only=True):
    system_filename = os.path.join(
        system_output_dir,
        f'xtreme_up_ner_{lang}_{model}-{checkpoint}.jsonl',
    )
    submission_filename = os.path.join(
        submission_dir, 'ner', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_semantic_parsing_files(
    system_output_dir: str, submission_dir: str, split: str, checkpoint: int
) -> None:
  """Defines the system output filenames and move them to the submission directory.
  """
  for lang in constants.get_languages(
      task='semantic_parsing', under_represented_only=True
  ):
    system_filename = (
        system_output_dir
        + '/'
        + f'xtreme_up_semantic_parsing_{lang}_{_MODEL.value}-{checkpoint}.jsonl'
    )
    submission_filename = os.path.join(
        submission_dir, 'semantic_parsing', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_asr_files(
    system_output_dir: str, submission_dir: str, split: str, checkpoint: int
) -> None:
  """Defines the system output filenames and move them to the submission directory.
  """
  for lang in constants.get_languages(task='asr', under_represented_only=True):
    system_filename = (
        system_output_dir
        + '/'
        + f'xtreme_up_asr_{lang}_{_MODEL.value}-{checkpoint}.jsonl'
    )
    submission_filename = os.path.join(
        submission_dir, 'asr', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def move_autocomplete_files(
    system_output_dir: str, submission_dir: str, split: str, checkpoint: int
) -> None:
  """Defines the system output filenames and move them to the submission directory."""
  for lang in constants.get_languages(
      'autocomplete', under_represented_only=True
  ):
    system_filename = (
        system_output_dir
        + '/'
        + f'xtreme_up_autocomplete_{lang}_{_MODEL.value}-{checkpoint}.jsonl'
    )
    submission_filename = os.path.join(
        submission_dir, 'autocomplete', split, f'{lang}.jsonl'
    )
    system_out_to_submission(system_filename, submission_filename)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tasks = _TASK.value.split(',')
  output_dirs = _SYSTEM_OUTPUT_DIR.value.split(',')
  checkpoints = _CHECKPOINT_STEP.value.split(',')
  split = _SPLIT.value

  for task, output_dir, checkpoint in zip(
      tasks, output_dirs, checkpoints, strict=True
  ):
    task = task.strip()
    checkpoint = int(checkpoint)

    if task == 'transliteration':
      move_transliteration_files(
          output_dir, _SUBMISSION_DIR.value, split, checkpoint
      )
    elif task == 'translation':
      move_mt_files(output_dir, _SUBMISSION_DIR.value, split, checkpoint)
    elif task == 'qa_in_lang':
      move_qa_in_language_files(
          output_dir, _SUBMISSION_DIR.value, split, checkpoint
      )
    elif task == 'qa_cross_lang':
      move_qa_cross_language_files(
          output_dir, _SUBMISSION_DIR.value, split, checkpoint
      )
    elif task == 'retrieval_in_lang':
      move_retrieval_in_language_files(output_dir, _SUBMISSION_DIR.value, split)
    elif task == 'retrieval_cross_lang':
      move_retrieval_cross_language_files(
          output_dir, _SUBMISSION_DIR.value, split
      )
    elif task == 'ner':
      move_ner_files(
          output_dir, _SUBMISSION_DIR.value, _MODEL.value, split, checkpoint
      )
    elif task == 'semantic_parsing':
      move_semantic_parsing_files(
          output_dir, _SUBMISSION_DIR.value, split, checkpoint
      )
    elif task == 'asr':
      move_asr_files(output_dir, _SUBMISSION_DIR.value, split, checkpoint)
    elif task == 'autocomplete':
      move_autocomplete_files(
          output_dir, _SUBMISSION_DIR.value, split, checkpoint
      )
    else:
      raise ValueError(f'{task} is not a task we support.')


if __name__ == '__main__':
  app.run(main)
