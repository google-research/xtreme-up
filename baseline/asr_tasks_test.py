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
"""Tests for XTREME-UP tasks."""

from absl.testing import absltest
from absl.testing import parameterized
import t5

from xtreme_up.asr import tasks as asr_tasks

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
_SEQUENCE_LENGTH = {'inputs': 128, 'targets': 128}

TASKS = asr_tasks.ALL_SPM_TASK_NAMES


class TasksTest(parameterized.TestCase):

  @parameterized.parameters(((name,) for name in TASKS))
  def test_task(self, name):
    task = TaskRegistry.get(name)
    self.assertIsNotNone(task)

    train_ds = task.get_dataset(_SEQUENCE_LENGTH, 'train').take(1)
    train_ds = list(train_ds.as_numpy_iterator())
    dev_ds = task.get_dataset(_SEQUENCE_LENGTH, 'validation').take(1)
    dev_ds = list(dev_ds.as_numpy_iterator())
    test_ds = task.get_dataset(_SEQUENCE_LENGTH, 'test').take(1)
    test_ds = list(test_ds.as_numpy_iterator())

    self.assertIn('inputs', train_ds[0])
    self.assertIn('targets', train_ds[0])
    self.assertIn('inputs', dev_ds[0])
    self.assertIn('targets', dev_ds[0])
    self.assertIn('inputs', test_ds[0])
    self.assertIn('targets', test_ds[0])


if __name__ == '__main__':
  absltest.main()
