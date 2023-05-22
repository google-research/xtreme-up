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
"""Seq2Seq semantic parsing tasks for the MTOP dataset."""

import collections
from collections.abc import Iterable
import seqio
import t5.data
import t5.evaluation.metrics as t5_metrics

from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants

ALL_TASK_NAMES = []


def _add_xtreme_up_semantic_parsing_tasks(model):
  """Adds XTREME-UP semantic parsing tasks."""
  tasks = collections.defaultdict(list)

  def get_mtop_preprocessors():
    return [
        t5.data.preprocessors.preprocess_tsv,
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim,
    ]

  for lang in constants.get_languages(task='semantic_parsing'):
    split_to_filepattern = tasks_lib.get_files_by_split(
        'semantic_parsing', lang
    )
    task_name = f'xtreme_up_semantic_parsing_{lang}_{model}'
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern),
        preprocessors=get_mtop_preprocessors(),
        output_features=tasks_lib.get_output_features(model),
        metric_fns=[t5_metrics.sequence_accuracy],
    )
    tasks['finetune'].append(task_name)
    ALL_TASK_NAMES.append(task_name)

  # Add finetuning tasks for all languages.
  seqio.MixtureRegistry.add(
      f'xtreme_up_semantic_parsing_{model}',
      tasks['finetune'],
      default_rate=1.0,
  )

  # Add finetuning tasks for all languages but code-switched data.
  seqio.MixtureRegistry.add(
      f'xtreme_up_semantic_parsing.{model}.finetune.nocs',
      [
          task_name
          for task_name in tasks['finetune']
          if '_cs' not in task_name
      ],
      default_rate=1.0,
  )

_add_xtreme_up_semantic_parsing_tasks('mt5')
_add_xtreme_up_semantic_parsing_tasks('byt5')
