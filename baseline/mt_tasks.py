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
"""MT tasks for XTREME-UP."""
import seqio
import t5.data

from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants
from xtreme_up.evaluation import metrics


ALL_SPM_TASK_NAMES = []
ALL_BYTE_TASK_NAMES = []

ALL_LOW_RESOURCE_SPM_TASK_NAMES = []
ALL_LOW_RESOURCE_BYTE_TASK_NAMES = []


for lang in constants.get_languages(task='translation'):
  lang_pair = f'en2{lang}'
  split_to_filepattern = tasks_lib.get_files_by_split('translation', lang_pair)
  task_name = f'xtreme_up_translation_en_{lang}'
  byte_task_name = f'{task_name}_byt5'
  seqio.TaskRegistry.add(
      task_name,
      source=seqio.TextLineDataSource(split_to_filepattern),
      preprocessors=[
          t5.data.preprocessors.preprocess_tsv,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=tasks_lib.get_output_features('mt5'),
      metric_fns=[metrics.bleu_seqio, metrics.chrf_seqio]
  )
  seqio.TaskRegistry.add(
      byte_task_name,
      source=seqio.TextLineDataSource(split_to_filepattern),
      preprocessors=[
          t5.data.preprocessors.preprocess_tsv,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=tasks_lib.get_output_features('byt5'),
      metric_fns=[metrics.bleu_seqio, metrics.chrf_seqio]
  )
  ALL_SPM_TASK_NAMES.append(task_name)
  ALL_BYTE_TASK_NAMES.append(byte_task_name)
  if constants.is_under_represented(lang):
    ALL_LOW_RESOURCE_SPM_TASK_NAMES.append(task_name)
    ALL_LOW_RESOURCE_BYTE_TASK_NAMES.append(byte_task_name)

seqio.MixtureRegistry.add(
    'xtreme_up_translation_all_langs_mt5', ALL_SPM_TASK_NAMES, default_rate=1.0)
seqio.MixtureRegistry.add(
    'xtreme_up_translation_mt5',  # By default, just fine-tune on ULs.
    ALL_LOW_RESOURCE_SPM_TASK_NAMES,
    default_rate=1.0,
)

seqio.MixtureRegistry.add(
    'xtreme_up_translation_all_langs_byt5',
    ALL_BYTE_TASK_NAMES,
    default_rate=1.0)
seqio.MixtureRegistry.add(
    'xtreme_up_translation_byt5',
    ALL_LOW_RESOURCE_BYTE_TASK_NAMES,
    default_rate=1.0)

# All tasks, to facilitate testing. Note that low-resource tasks are a strict
# subset of these.
ALL_TASK_NAMES = ALL_SPM_TASK_NAMES + ALL_BYTE_TASK_NAMES
