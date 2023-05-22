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
"""Tests for NER tasks."""
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import t5

from xtreme_up.ner import tasks

TaskRegistry = t5.data.TaskRegistry
MixtureRegistry = t5.data.MixtureRegistry
_SEQUENCE_LENGTH = {'inputs': 128, 'targets': 128}

ner_tasks = []
for model in ['mt5', 'byt5']:
  for dataset in ['masakhaner', 'masakhaner_2.0', 'masakhaner_x']:
    for lang in tasks.DATASET_LANG_DICT[dataset]:
      ner_tasks.append(f'{model}.{dataset}.{lang}')


class TasksTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in ner_tasks))
  def test_task(self, name):
    task = TaskRegistry.get(name)
    self.assertIsNotNone(task)

    ds = task.get_dataset(_SEQUENCE_LENGTH, 'train').take(5)
    ds = list(ds.as_numpy_iterator())
    logging.info(ds)

    self.assertIn('inputs', ds[0])
    self.assertIn('targets', ds[0])

if __name__ == '__main__':
  absltest.main()
