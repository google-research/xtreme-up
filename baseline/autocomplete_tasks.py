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
"""Autocomplete tasks and mixtures for XTREME-UP."""
import collections
from typing import Any

import seqio
import t5.data
import tensorflow as tf

from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants


TOP_K_SIZE = 3


def gather_predictions(
    target_or_pred: list[str],
    example: dict[str, tf.Tensor],
    is_target: bool,
) -> list[Any]:
  if is_target:
    return [example["targets_pretokenized"] for _ in range(TOP_K_SIZE)]
  else:
    return target_or_pred


# A metric is needed during finetuning to produce
# inference_eval/metrics.jsonl files for each language. These files are compiled
# in a inference_eval/summary.csv during eval. This is in turn used to loop over
# all checkpoints to select the best one. However the metric on seqio always
# return 0. when we use multiple decodes. This prevent automatic checkpoint
# selection with multiple decodes.
def dummy_metric(
    targets: list[str], predictions: list[str]
) -> dict[str, float]:
  assert len(targets) == len(predictions)
  return {"dummy_metric": 0.0}


_model_tasks = collections.defaultdict(list)
for model in ('mt5', 'byt5'):
  for lang in constants.get_languages(task="autocomplete"):
    split_to_filepattern = tasks_lib.get_files_by_split("autocomplete", lang)
    task_name = f"xtreme_up_autocomplete_{lang}_{model}"
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
        postprocess_fn=gather_predictions,
        metric_fns=[dummy_metric],
    )
    _model_tasks[model].append(task_name)

  seqio.MixtureRegistry.add(
      f"xtreme_up_autocomplete_{model}", _model_tasks[model], default_rate=1.0
  )
