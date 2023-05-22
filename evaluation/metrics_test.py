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
"""Tests for metrics."""
from typing import Literal
from absl.testing import absltest

from xtreme_up.evaluation import metrics


class SimpleCerTest(absltest.TestCase):

  def testTxtPreprocess(self):
    txt = 'abcd [ABC] (fg) ???   OK! '
    txt_preprocessed = metrics.text_preprocess(txt)
    self.assertEqual(txt_preprocessed, 'abcd abc fg ok')

  def testUnicodeNormalizationKorean(self):
    txt = '읶'  # length of 1
    nfkc_txt: Literal['읶'] = '읶'  # length of 1

    self.assertLen(txt, 1)

    # NFKC normalization keeps the length for Korean.
    self.assertEqual(nfkc_txt, metrics.unicode_normalization(txt))
    self.assertLen(nfkc_txt, 1)

  def testCerLatinDeletion(self):
    ref = ['this is a cat']
    hyp = ['this is a t']
    cer = metrics.cer(
        targets=ref, predictions=hyp, apply_asr_normalization=True
    )
    self.assertAlmostEqual(cer, 15.384615384615385)

  def testCleaningPunctuation(self):
    # The puncuation will be removed by apply_asr_normalization before
    # calculating the CER.
    ref = ['this is a cat?']
    hyp = ['this is a t']
    cer = metrics.cer(
        targets=ref, predictions=hyp, apply_asr_normalization=True
    )
    self.assertAlmostEqual(cer, 15.384615384615385)

  def testCleaningMultipleExtraSpaceLatin(self):
    # The multiple extra space will be truncated to one space.
    ref = ['this is a     cat']
    hyp = ['this is a t']
    cer = metrics.cer(
        targets=ref, predictions=hyp, apply_asr_normalization=True
    )
    self.assertAlmostEqual(cer, 15.384615384615385)

  def testCerCjkSubstitution(self):
    ref = ['如果有多一張船票']
    hyp = ['如果冇多一張船票']
    cer = metrics.cer(
        targets=ref, predictions=hyp, apply_asr_normalization=True
    )
    self.assertAlmostEqual(cer, 12.5)

  def testCleaningSingleSpace(self):
    ref = ['如果 有多一張船票']
    hyp = ['如果冇多一張船票']
    cer = metrics.cer(
        targets=ref, predictions=hyp, apply_asr_normalization=True
    )
    self.assertAlmostEqual(cer, 22.22222222222222)

  def testCleaningMultipleExtraSpace(self):
    ref = ['如果  有多一張船票']
    hyp = ['如果冇多一張船票']
    cer = metrics.cer(
        targets=ref, predictions=hyp, apply_asr_normalization=True
    )
    self.assertAlmostEqual(cer, 22.22222222222222)

  def testCerLatinInsertion(self):
    ref = ['mucho gusto']
    hyp = ['muchoagusto']
    cer = metrics.cer(
        targets=ref, predictions=hyp, apply_asr_normalization=True
    )
    self.assertAlmostEqual(cer, 9.090909090909091)


if __name__ == '__main__':
  absltest.main()
