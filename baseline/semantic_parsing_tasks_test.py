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

from xtreme_up.semantic_parsing import tasks as sp_tasks

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
_SEQUENCE_LENGTH = {'inputs': 128, 'targets': 128}

TASKS = sp_tasks.ALL_TASK_NAMES


class TasksTest(parameterized.TestCase):

  def _test_task_inputs_targets(self, task_name: str, split: str) -> None:
    task = TaskRegistry.get(task_name)
    self.assertIsNotNone(task)
    ds = task.get_dataset(_SEQUENCE_LENGTH, split).take(1)
    ds = list(ds.as_numpy_iterator())
    self.assertIn('inputs', ds[0])
    self.assertIn('targets', ds[0])

  @parameterized.parameters(((name,) for name in TASKS))
  def test_task(self, name: str) -> None:
    self._test_task_inputs_targets(task_name=name, split='train')
    self._test_task_inputs_targets(task_name=name, split='validation')
    self._test_task_inputs_targets(task_name=name, split='test')


if __name__ == '__main__':
  absltest.main()
