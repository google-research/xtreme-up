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
"""Selects file I/O functionality for either Google internal or open-source usage.
"""
import os


def open(path: str, mode: str):  # pylint: disable=redefined-builtin
  # Use TensorFlow's I/O routines if available, otherwise fall back to native
  # Python I/O.
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    return tf.io.gfile.GFile(path, mode)
  except:  # pylint: disable=bare-except
    return open(path, mode)


def listdir(path: str):  # pylint: disable=redefined-builtin
  # Use TensorFlow's I/O routines if available, otherwise fall back to native
  # Python I/O.
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    return tf.io.gfile.listdir(path)
  except:  # pylint: disable=bare-except
    return os.listdir(path)


def join(paths: list[str]):  # pylint: disable=redefined-builtin
  # Use TensorFlow's I/O routines if available, otherwise fall back to native
  # Python I/O.
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    return tf.io.gfile.join(*paths)
  except:  # pylint: disable=bare-except
    return os.path.join(*paths)


def exists(path: str) -> bool:  # pylint: disable=redefined-builtin
  # Use TensorFlow's I/O routines if available, otherwise fall back to native
  # Python I/O.
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    return tf.io.gfile.exists(path)
  except:  # pylint: disable=bare-except
    return os.path.exists(path)


def makedirs(path: str) -> bool:  # pylint: disable=redefined-builtin
  # Use TensorFlow's I/O routines if available, otherwise fall back to native
  # Python I/O.
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    return tf.io.gfile.makedirs(path)
  except:  # pylint: disable=bare-except
    return os.makedirs(path)
