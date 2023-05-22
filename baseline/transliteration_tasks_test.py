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
"""Tests for XTREME-UP transliteration tasks."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import t5
import tensorflow.compat.v1 as tf

from xtreme_up.transliteration import tasks  # pylint:disable=unused-import

tf.disable_v2_behavior()
tf.enable_eager_execution()

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
_SEQUENCE_LENGTH = {"inputs": 128, "targets": 128}

_TASKS = [
    "byt5_dakshina_full_string_translit.am_Latn_Ethi",
    "byt5_dakshina_full_string_translit.bn_Latn_Beng",
    "byt5_dakshina_full_string_translit.pa_Arab_Guru",
    "byt5_dakshina_full_string_translit.pa_Arab_Latn",
    "byt5_dakshina_full_string_translit.pa_Guru_Arab",
    "byt5_dakshina_full_string_translit.ur_Arab_Latn",
    "mt5_dakshina_full_string_translit.gu_Latn_Gujr",
    "mt5_dakshina_full_string_translit.kn_Knda_Latn",
    "mt5_dakshina_full_string_translit.pa_Guru_Latn",
    "mt5_dakshina_full_string_translit.pa_Latn_Arab",
]

_MIXTURES = [
    "byt5_dakshina_full_string_translit",
    "mt5_dakshina_full_string_translit",
]


class TransliterationTasksTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in _TASKS))
  def test_task(self, name):
    task = TaskRegistry.get(name)
    split = "train" if "train" in task.splits else "validation"
    logging.info("task=%s, split=%s", name, split)
    ds = task.get_dataset(_SEQUENCE_LENGTH, split)
    for d in ds:
      logging.info(d)
      break

  @parameterized.parameters(((name,) for name in _MIXTURES))
  def test_mixture(self, name):
    mixture = MixtureRegistry.get(name)
    logging.info("mixture=%s", name)
    ds = mixture.get_dataset(_SEQUENCE_LENGTH, "train")
    for d in ds:
      logging.info(d)
      break


if __name__ == "__main__":
  absltest.main()
