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
import seqio
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics
from t5.evaluation import qa_utils

from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants


DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

def squad(targets, predictions):
  """Computes SQuAD metrics - F1 and EM.

  Args:
    targets: list of lists of strings
    predictions: list of strings

  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [[qa_utils.normalize_squad(t)] for t in targets]
  predictions = [qa_utils.normalize_squad(p) for p in predictions]
  return qa_utils.qa_metrics(targets, predictions)


def boolean_qa(targets, predictions):
  """Boolean accuracy metric."""
  unanswerable = 'No Answer'
  answerable = 'Has Answer'
  binarized_targets = targets[:]
  binarized_predictions = predictions[:]
  for i in range(len(targets)):
    if targets[i] != unanswerable:
      binarized_targets[i] = answerable
    if predictions[i] != unanswerable:
      binarized_predictions[i] = answerable
  accuracy = t5_metrics.accuracy(binarized_targets, binarized_predictions)[
      'accuracy'
  ]
  return {'boolean_accuracy': accuracy}


for model in ('mt5', 'byt5'):
  # In-language QA Finetuning Tasks:
  task_names = []
  for lang in constants.get_languages(task='qa_in_lang'):
    task_name = f'xtreme_up_qa_in_lang.{lang}_{model}'
    split_to_filepattern = tasks_lib.get_files_by_split('qa_in_lang', lang)
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern),
        preprocessors=[t5_preprocessors.preprocess_tsv] + DEFAULT_PREPROCESSORS,
        output_features=tasks_lib.get_output_features(model),
        metric_fns=[squad, boolean_qa])
    task_names.append(task_name)

  # In-language QA Finetuning Mixtures:
  seqio.MixtureRegistry.add(
      f'xtreme_up_qa_in_lang_{model}', task_names, default_rate=1.0)

  # Cross-language QA Finetuning Tasks:
  task_names = []
  for lang in constants.get_languages(task='qa_cross_lang'):
    task_name = f'xtreme_up_qa_cross_lang.{lang}_{model}'
    split_to_filepattern = tasks_lib.get_files_by_split('qa_cross_lang', lang)
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern),
        preprocessors=[t5_preprocessors.preprocess_tsv] + DEFAULT_PREPROCESSORS,
        output_features=tasks_lib.get_output_features(model),
        metric_fns=[squad, boolean_qa],
    )
    task_names.append(task_name)

  # Cross-language QA Finetuning Mixtures:
  seqio.MixtureRegistry.add(
      f'xtreme_up_qa_cross_lang_{model}', task_names, default_rate=1.0)
