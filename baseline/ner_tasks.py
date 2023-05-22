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
"""NER tasks for XTREME-UP."""
import seqio
from t5.data import preprocessors as t5_preprocessors

from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants
from xtreme_up.evaluation import metrics


for model in ['byt5', 'mt5']:
  model_tasks = []
  for language in constants.get_languages(task='ner'):
    split_to_filepattern = tasks_lib.get_files_by_split('ner', language)
    task_name = f'xtreme_up_ner_{language}_{model}'

    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(
            split_to_filepattern, skip_header_lines=0
        ),
        preprocessors=[
            t5_preprocessors.preprocess_tsv,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=tasks_lib.get_output_features(model),
        metric_fns=[metrics.span_f1_seqio],
    )

    model_tasks.append(task_name)

  seqio.MixtureRegistry.add(
      f'xtreme_up_ner_{model}', model_tasks, default_rate=1.0
  )
