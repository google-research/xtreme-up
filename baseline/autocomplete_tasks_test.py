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
"""Tests for XTREME-UP Autocomplete tasks."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import seqio
import t5
import tensorflow.compat.v1 as tf

from xtreme_up.baseline import autocomplete_tasks  # pylint:disable=unused-import
from xtreme_up.evaluation import constants


tf.disable_v2_behavior()
tf.enable_eager_execution()

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
_SEQUENCE_LENGTH = {"inputs": 128, "targets": 16}

_TASKS = [
    "autocomplete.mt5" + "." + lang
    for lang in constants.get_languages(task="autocomplete")
]
_TASKS += [
    "autocomplete.byt5" + "." + lang
    for lang in constants.get_languages(task="autocomplete")
]

_MIXTURES = ["autocomplete.mt5_mixture", "autocomplete.byt5_mixture"]

class AutocompleteTaskTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in _TASKS))
  def test_task(self, name):
    local_task = TaskRegistry.get(name)
    split = "train" if "train" in local_task.splits else "validation"
    logging.info("task=%s, split=%s", name, split)
    dataset = local_task.get_dataset(_SEQUENCE_LENGTH, split=split)
    dataset = list(dataset.take(3).as_numpy_iterator())
    self.assertIn("inputs", dataset[0])
    self.assertIn("targets", dataset[0])
    for ex in dataset:
      print(ex)

  def test_metrics(self):
    evaluator = seqio.Evaluator(
        mixture_or_task_name="autocomplete.mt5.en_",
        feature_converter=seqio.EncDecFeatureConverter(pack=False),
        eval_split="validation",
    )
    metrics, _ = evaluator.evaluate(
        compute_metrics=True,
        model_fns=None,
        step=None,
        predict_fn=None,
        score_fn=None)
    print(metrics.result())

  @parameterized.parameters(((name,) for name in _MIXTURES))
  def test_mixture(self, name):
    mixture = MixtureRegistry.get(name)
    split = "train" if "train" in mixture.splits else "validation"
    logging.info("mixture=%s", name)
    dataset = mixture.get_dataset(_SEQUENCE_LENGTH, split=split)
    dataset = list(dataset.take(3).as_numpy_iterator())
    self.assertIn("inputs", dataset[0])
    self.assertIn("targets", dataset[0])
    for ex in dataset:
      print(ex)


if __name__ == "__main__":
  absltest.main()
