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
"""ASR tasks for XTREME-UP."""
import seqio
import t5.data
from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants
from xtreme_up.evaluation import metrics


ALL_SPM_TASK_NAMES = []
ALL_BYTE_TASK_NAMES = []

BYTE_VOCAB = seqio.ByteVocabulary()
BYTE_OUTPUT_FEATURES = {
    'inputs': t5.data.Feature(
        vocabulary=BYTE_VOCAB, add_eos=True, required=False
    ),
    'targets': t5.data.Feature(vocabulary=BYTE_VOCAB, add_eos=True),
}

for lang in constants.get_languages(task='asr', under_represented_only=True):
  split_to_filepattern = tasks_lib.get_files_by_split(task='asr', lang=lang)
  task_name = f'xtreme_up_asr_{lang}_mt5'
  byte_task_name = task_name.replace('mt5', 'byt5')

  seqio.TaskRegistry.add(
      task_name,
      source=seqio.TextLineDataSource(
          split_to_filepattern, skip_header_lines=0
      ),
      preprocessors=[
          t5.data.preprocessors.preprocess_tsv,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=tasks_lib.DEFAULT_OUTPUT_FEATURES,
      metric_fns=[metrics.cer_seqio],
  )
  seqio.TaskRegistry.add(
      byte_task_name,
      source=seqio.TextLineDataSource(
          split_to_filepattern, skip_header_lines=0
      ),
      preprocessors=[
          t5.data.preprocessors.preprocess_tsv,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=BYTE_OUTPUT_FEATURES,
      metric_fns=[metrics.cer_seqio],
  )
  ALL_SPM_TASK_NAMES.append(task_name)
  ALL_BYTE_TASK_NAMES.append(byte_task_name)

seqio.MixtureRegistry.add(
    'xtreme_up_asr_mt5', ALL_SPM_TASK_NAMES, default_rate=1.0
)

seqio.MixtureRegistry.add(
    'xtreme_up_asr_byt5',
    ALL_BYTE_TASK_NAMES,
    default_rate=1.0,
)
