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
"""Implementation of various evaluation metrics for the XTREME-UP benchmark."""

import collections
import json
import re
from typing import Mapping, Union
import unicodedata
from absl import logging
import nltk
import numpy as np
import sacrebleu

from xtreme_up.evaluation import file_utils
from xtreme_up.evaluation import qa_utils


def text_preprocess(text: str) -> str:
  """Preprocess text before CER caculation."""

  # Lowercase, remove \t and new line.
  text = re.sub(r"[\t\n]", " ", text.lower())

  # Remove punctuation before space.
  text = re.sub(r"[,.\?!]+ ", " ", text)

  # Remove punctuation before end.
  text = re.sub(r"[,.\?!]+$", " ", text)

  # Remove punctuation after space.
  text = re.sub(r" [,.\?!]+", " ", text)

  # Remove quotes, [, ], ( and ).
  text = re.sub(r"['\(\)\[\]]", " ", text)

  # Remove extra space.
  text = re.sub(" +", " ", text.strip())
  return text


def unicode_normalization(text: str) -> str:
  """Applies Unicode normalization."""
  return unicodedata.normalize("NFKC", text)


def sequence_accuracy(targets: list[str], predictions: list[str]) -> float:
  """Computes per-sequence accuracy.

  This function is copied from t5.evaluation.metrics
  For each example, returns 1.0 if the target sequence EXACTLY matches the
  predicted sequence. Else, 0.0.

  Args:
    targets: list of strings
    predictions: list of strings

  Returns:
    float. Average sequence-level accuracy.
  """
  assert len(targets) == len(predictions)
  seq_acc = 100 * np.mean(
      [p == t for p, t in zip(predictions, targets, strict=True)]
  )
  return seq_acc


def chrf(targets: list[str], predictions: list[str]) -> float:
  """Computes chrF score from https://aclanthology.org/W15-3049/."""
  return 100 * sacrebleu.corpus_chrf(predictions, targets)


def chrf_seqio(targets: list[str], predictions: list[str]) -> dict[str, float]:
  """A SeqIO compatible wrapper around our internal chrF implementation."""
  return {"chrf": chrf(targets, predictions)}


def has_match(target: str, topk: list[str]) -> bool:
  """Do any of the top-k match the target?"""
  return any(pred == target for pred in topk)


def top3_accuracy(targets: list[str], predictions: list[list[str]]) -> float:
  """Computes Acc@3 with top-3 decode format."""
  for topk in predictions:
    if len(topk) != 3:
      raise ValueError(
          f"Expected exactly top-3 for all examples, but found ({len(topk)})" +
          str(topk)
      )
  sentence_level_acc = [
      has_match(target, topk)
      for target, topk in zip(targets, predictions, strict=True)
  ]
  return 100 * np.mean(sentence_level_acc)


def cer(
    targets: list[str],
    predictions: list[str],
    apply_asr_normalization: bool = False,
) -> float:
  """Computes the Character Error Rate (CER).

  This function is copied from
  https://github.com/google-research/byt5/blob/master/byt5/metrics.py

  The Character Error Rate for a (input word, target word) pair is defined as
  the minimum number of edits required to transform the input word to the target
  word divided by the total number of characters in the target word. The minimum
  number of edits is calculated using Levenshtein distance, where any
  single-character edit (insertion, deletion, substitution) is allowed.
  The CER for a list of (input word, target word) pairs is defined as the total
  number of edits required to transform each input word into the corresponding
  target word divided by the sum of the number of characters in the target
  words. For example, given:
    targets = ["abc", "aa"]
    predictions = ["abd", "a"]
  the CER would be: (1 + 1) / (3 + 2) = 2 / 5 = 0.4

  Args:
    targets: list of gold targets.
    predictions: list of model outputs.
    apply_asr_normalization: to apply text preprocessing used by ASR baseline.

  Returns:
    float, CER value for the predictions compared to the targets.
  """
  total_characters = 0
  total_edit_distance = 0
  if apply_asr_normalization:
    predictions = [text_preprocess(pred) for pred in predictions]
    targets = [text_preprocess(tgt) for tgt in targets]
  predictions = [unicode_normalization(pred) for pred in predictions]
  targets = [unicode_normalization(tgt) for tgt in targets]
  for target, prediction in zip(targets, predictions, strict=True):
    total_edit_distance += nltk.edit_distance(target, prediction)
    total_characters += len(target)

  return 100 * float(total_edit_distance) / total_characters


def cer_seqio(
    targets: list[str], predictions: list[str]
) -> Mapping[str, float]:
  """A SeqIO compatible wrapper around our internal CER implementation."""
  return {"cer": cer(targets, predictions)}


def bleu(
    targets: Union[list[str], list[list[str]]],
    predictions: list[str],
    tokenizer: str = "intl",
) -> float:
  """Computes BLEU score.

  This function is copied from
  https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/metrics.py

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings
    tokenizer: tokenizer option for corpus_bleu

  Returns:
    bleu_score across all targets and predictions
  """
  if isinstance(targets[0], list):
    targets = [list(target) for target in targets]
  else:
    # Need to wrap targets in another list for corpus_bleu.
    targets = [targets]

  # `corpus_bleu` returns a `sacrebleu.metrics.BLEUScore`.
  return (
      100
      * sacrebleu.corpus_bleu(
          predictions,
          targets,
          smooth_method="exp",
          smooth_value=0.0,
          force=False,
          lowercase=False,
          tokenize=tokenizer,
          use_effective_order=False,
      ).score
  )


def bleu_seqio(
    targets: Union[list[str], list[list[str]]],
    predictions: list[str],
    tokenizer: str = "intl",
) -> Mapping[str, float]:
  """A SeqIO compatible wrapper around our internal BLEU implementation."""
  return {"bleu": bleu(targets, predictions, tokenizer)}


def f1(targets: list[str], predictions: list[str]) -> float:
  """Computes token level F1 score."""
  targets = [qa_utils.normalize_squad(t) for t in targets]
  predictions = [qa_utils.normalize_squad(p) for p in predictions]
  return qa_utils.qa_metrics(targets, predictions)["f1"]


def span_f1_seqio(targets, predictions):
  """Computes Span based F1 score.

  This function is copied from
  https://github.com/google-research/multilingual-t5/blob/master/multilingual_t5/evaluation/metrics.py

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

  Returns:
    span f1 across all targets and predictions (Based on CoNLL script)
  """
  true_positives = collections.defaultdict(int)
  false_positives = collections.defaultdict(int)
  false_negatives = collections.defaultdict(int)

  def tags_to_spans(tag_sequence, delimiter=" $$ "):
    """Extract spans from IOB1 or BIO tags."""
    tag_sequence_split = [x.strip() for x in tag_sequence.split(delimiter)]
    tags_entities = []
    for tag_entity in tag_sequence_split:
      tag_entity_split = tag_entity.split(":")
      if len(tag_entity_split) != 2:
        continue
      tag = tag_entity_split[0].strip()
      entity = tag_entity_split[1].strip()
      tags_entities.append((tag, entity))
    return tags_entities

  def compute_f1_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(
        true_positives + false_positives + 1e-13
    )
    recall = float(true_positives) / float(
        true_positives + false_negatives + 1e-13
    )
    f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure

  for target, pred in zip(targets, predictions, strict=True):
    gold_spans = tags_to_spans(target)
    predicted_spans = tags_to_spans(pred)

    for span in predicted_spans:
      if span in gold_spans:
        true_positives[span[0]] += 1
        gold_spans.remove(span)
      else:
        false_positives[span[0]] += 1
    # These spans weren't predicted.
    for span in gold_spans:
      false_negatives[span[0]] += 1

  _, _, f1_measure = compute_f1_metrics(
      sum(true_positives.values()),
      sum(false_positives.values()),
      sum(false_negatives.values()),
  )

  return {"span_f1": f1_measure}


def span_f1(targets: list[str], predictions: list[str]) -> float:
  """Computes span F1 score based on mT5/ByT5 output format."""
  return 100 * span_f1_seqio(targets, predictions)["span_f1"]


def compute_mrr(targets: list[str], predictions: list[list[str]]) -> float:
  """Compute Mean Reciprocal Rank at 10."""
  mrr_total = 0

  for target, predicted_neighbors in zip(targets, predictions, strict=True):
    rank = 0
    for i, neighbor in enumerate(predicted_neighbors):
      if target == neighbor:
        # Matched the target at this position in the ranking.
        rank = i + 1
        break
    if rank > 0:
      mrr_total += 1 / rank

  return mrr_total / len(targets)


def parse_jsonl_gold_file(
    gold_jsonl_file: str, target_field: str = "target"
) -> list[str]:
  """Gets the gold targets from the data jsonl file."""
  logging.info("Reading results from %s", gold_jsonl_file)
  targets = []
  with file_utils.open(gold_jsonl_file, mode="r") as f:
    for line in f:
      line = line.strip()
      if not line:  # Skip blank lines.
        continue
      data = json.loads(line)
      targets.append(data[target_field])
  return targets


def parse_jsonl_prediction_file(
    prediction_jsonl_file: str,
) -> list[str] | list[list[str]]:  # Nested list for autocomplete + retrieval.
  """Gets the predictions from a jsonl file."""
  logging.info("Reading results from %s", prediction_jsonl_file)
  predictions = []
  with file_utils.open(prediction_jsonl_file, mode="r") as f:
    for line in f:
      line = line.strip()
      if not line:  # Skip blank lines.
        continue
      data = json.loads(line)
      predictions.append(data["prediction"])
  return predictions


def score_file(
    gold_jsonl_file: str,
    prediction_jsonl_file: str,
    metric_name: str,
) -> float:
  """Computes the score for a given result file."""

  # First, we get the predictions and gold targets.
  predictions = parse_jsonl_prediction_file(prediction_jsonl_file)
  targets = parse_jsonl_gold_file(
      gold_jsonl_file,
      # Retrieval uses 'id' as its gold target field.
      target_field="target" if metric_name != "mrr" else "id",
  )

  # Handle MRR for retrieval as a special case since it's slightly different.
  if len(targets) != len(predictions):
    raise ValueError(
        f"Found {len(targets)} targets and {len(predictions)} predictions, but "
        "expected counts to match. "
        f"targets={gold_jsonl_file} predictions={prediction_jsonl_file}"
    )

  # Next, we run the corresponding eval.
  if metric_name.lower() == "cer":
    result = cer(targets, predictions)
  elif metric_name.lower() == "cer_normalized":
    result = cer(targets, predictions, apply_asr_normalization=True)
  elif metric_name.lower() == "chrf":
    result = chrf(targets, predictions)
  elif metric_name.lower() == "f1":
    result = f1(targets, predictions)
  elif metric_name.lower() == "span_f1":
    result = span_f1(targets, predictions)
  elif metric_name.lower() == "sequence_accuracy":
    result = sequence_accuracy(targets, predictions)
  elif metric_name.lower() == "top3_accuracy":
    result = top3_accuracy(targets, predictions)
  elif metric_name.lower() == "mrr":
    result = compute_mrr(targets, predictions)
  else:
    raise ValueError("Error: {} is not a supported metric!".format(metric_name))
  return result  # pytype: disable=name-error  # py310-upgrade
