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
"""Transliteration tasks and mixtures for XTREME-UP."""
import collections

import seqio
import t5.data

from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants
from xtreme_up.evaluation import metrics


# ==============================================================================
# Full-string Transliteration Finetuning Tasks for original Dakshina data.
# ==============================================================================
# Dakshina: https://github.com/google-research-datasets/dakshina
#
# In addition to the original data, Amharic has been added.

# Finetuning Tasks:
_model_tasks = collections.defaultdict(list)
for model in ('mt5', 'byt5'):
  for lang in constants.TRANSLIT_LANGS_AND_SCRIPTS:
    for script_1, script_2 in constants.TRANSLIT_LANGS_AND_SCRIPTS[lang]:
      # Task for full-string transliteration: Latin -> Native direction for
      # the 1-st inner loop run and Native -> Latin direction for the 2-d run
      # for the most languages (things are slightly more complicated for Punjabi
      # because we have the data for two scripts).
      #
      # These tasks are meant to fine-tune all the languages together.
      # The input features (in Latin script for 1-st inner loop run and in
      # Native script for the 2-d run) are prefixed with the BCP-47
      # language and ISO 15924 source-target script codes.
      for src_script, tgt_script in (
          (script_1, script_2),
          (script_2, script_1),
      ):
        task_name = f"xtreme_up_transliteration_{lang}_{src_script}_{tgt_script}_{model}"
        split_to_filepattern = tasks_lib.get_files_by_split(
            "transliteration", f"{src_script}2{tgt_script}.{lang}"
        )
        seqio.TaskRegistry.add(
            task_name,
            source=seqio.TextLineDataSource(split_to_filepattern),
            preprocessors=[
                t5.data.preprocessors.preprocess_tsv,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            output_features=tasks_lib.get_output_features(model),
            metric_fns=[metrics.cer_seqio],
        )
        _model_tasks[model].append(task_name)

  # Finetuning Mixtures:
  seqio.MixtureRegistry.add(
      f"xtreme_up_transliteration_{model}",
      _model_tasks[model],
      default_rate=1.0,
  )
