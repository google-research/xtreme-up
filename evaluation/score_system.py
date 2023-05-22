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
"""Regenerates the XTREME-UP public results table."""

from collections.abc import Sequence
import dataclasses
import json
import os
from typing import Any, Callable, Optional, Union

from absl import app
from absl import flags
from absl import logging

from xtreme_up.evaluation import constants
from xtreme_up.evaluation import file_utils
from xtreme_up.evaluation import metrics

_SUBMISSION_NAME = flags.DEFINE_string(
    'submission_name',
    None,
    (
        'The unique name for a submission. It is also the folder name'
        ' containing all the outputs.'
    ),
)

_SUBMISSION_FOLDER = flags.DEFINE_string(
    'submission_folder',
    None,
    'The path to the folder containing all the outputs.',
)

_GOLD_DATA_FOLDER = flags.DEFINE_string(
    'gold_data_folder',
    None,
    'The path to the folder containing the gold test data.',
)

_JSON_OUT = flags.DEFINE_string(
    'json_out',
    None,
    'A file to write the evaluation result for a single submission.',
    required=True,
)

_SCORE_TASK_NAME = flags.DEFINE_string(
    'score_task_name',
    None,
    'The name of the task to score. Score all tasks if not set.',
    required=False,
)


@dataclasses.dataclass
class Task:
  """Definition of an XTREME-UP task for use in reporting eval results."""

  # on a per-task basis in this file.
  full_name: str
  simple_name: str
  lang: str
  metric: str
  output_file_name: str
  gold_file_name: str


@dataclasses.dataclass
class TaskMetadata:
  """Metadata for a task in XTREME-UP."""

  match_pattern: str
  simple_name: str
  metric: Union[str, Callable[[str], str]]
  file_configs: Any

  def match(self, task_name: str) -> bool:
    return self.match_pattern in task_name


XTREME_UP_UNDER_REPRESENTED_LANGS = frozenset([
    'am', 'as', 'ast', 'az', 'bbj', 'bem', 'ber', 'bm', 'bo', 'ckb', 'cy',
    'ee', 'ff', 'fon', 'ga', 'gd', 'gu', 'ha', 'hsb', 'hy', 'ig', 'is', 'jv',
    'kam', 'kea', 'km', 'kn', 'ky', 'lb', 'lg', 'lij', 'ln', 'lo', 'luo',
    'mg', 'mi', 'mk', 'ml', 'mn', 'mos', 'my', 'nd', 'ne', 'nso', 'nw', 'ny',
    'oc', 'olo', 'om', 'pa', 'ps', 'sa', 'sd', 'se', 'si', 'sn', 'so', 'ss',
    'st', 'sw', 'te', 'tg', 'tn', 'ts', 'tw', 'umb', 've', 'wo', 'xh', 'yo',
    'zu'
])

_TRANSLITERATION_METADATA = TaskMetadata(
    match_pattern='transliteration',
    simple_name='transliteration',
    metric='CER',
    file_configs={
        'am': [('Latn', 'Ethi')],
        'bn': [('Latn', 'Beng')],
        'gu': [('Latn', 'Gujr')],
        'hi': [('Latn', 'Deva')],
        'kn': [('Latn', 'Knda')],
        'ml': [('Latn', 'Mlym')],
        'mr': [('Latn', 'Deva')],
        # Gurmukhi (Guru) and Shahmukhi (Arab) for Punjabi.
        'pa': [('Latn', 'Guru'), ('Arab', 'Guru'), ('Latn', 'Arab')],
        'sd': [('Latn', 'Arab')],
        'si': [('Latn', 'Sinh')],
        'ta': [('Latn', 'Taml')],
        'te': [('Latn', 'Telu')],
        'ur': [('Latn', 'Arab')],
    },
)

_MT_METADATA = TaskMetadata(
    match_pattern='translation',
    simple_name='translation',
    metric='ChrF',
    file_configs=constants.get_languages(
        'translation', under_represented_only=True
    ),
)

_QA_IN_LANG_METADATA = TaskMetadata(
    match_pattern='qa_in_lang',
    simple_name='qa_in_lang',
    metric='f1',
    file_configs=constants.get_languages(
        'qa_in_lang', under_represented_only=True
    ),
)

_QA_CROSS_LANG_METADATA = TaskMetadata(
    match_pattern='qa_cross_lang',
    simple_name='qa_cross_lang',
    metric='f1',
    file_configs=constants.get_languages(
        'qa_cross_lang', under_represented_only=True
    ),
)

_RETRIEVAL_IN_LANG_METADATA = TaskMetadata(
    match_pattern='retrieval_in_lang',
    simple_name='retrieval_in_lang',
    metric='mrr',
    file_configs=constants.get_languages(
        'retrieval_in_lang', under_represented_only=True
    ),
)

_RETRIEVAL_CROSS_LANG_METADATA = TaskMetadata(
    match_pattern='retrieval_cross_lang',
    simple_name='retrieval_cross_lang',
    metric='mrr',
    file_configs=constants.get_languages(
        'retrieval_cross_lang', under_represented_only=True
    ),
)

_NER_METADATA = TaskMetadata(
    match_pattern='ner',
    simple_name='ner',
    metric='Span_F1',
    file_configs=constants.get_languages('ner', under_represented_only=True),
)

_SEMANTIC_PARSING_METADATA = TaskMetadata(
    match_pattern='semantic_parsing',
    simple_name='semantic_parsing',
    metric='sequence_accuracy',
    file_configs=constants.get_languages(
        'semantic_parsing', under_represented_only=True
    ),
)

_ASR_METADATA = TaskMetadata(
    match_pattern='asr',
    simple_name='asr',
    metric='CER_normalized',
    file_configs=constants.get_languages('asr', under_represented_only=True),
)

_AUTOCOMPLETE_METADATA = TaskMetadata(
    match_pattern='autocomplete',
    simple_name='autocomplete',
    metric='top3_accuracy',
    file_configs=constants.get_languages(
        'autocomplete', under_represented_only=True
    ),
)

_ALL_METADATA = [
    _MT_METADATA,
    _QA_IN_LANG_METADATA,
    _QA_CROSS_LANG_METADATA,
    _RETRIEVAL_IN_LANG_METADATA,
    _RETRIEVAL_CROSS_LANG_METADATA,
    _NER_METADATA,
    _TRANSLITERATION_METADATA,
    _SEMANTIC_PARSING_METADATA,
    _ASR_METADATA,
    _AUTOCOMPLETE_METADATA,
]

_ALL_SUB_TASK_FILES: list[tuple[str, str]] = []
_ALL_GOLD_DATA_FILES: list[str] = []

# Add filenames for each task.
# Transliteration
for lang, scripts in _TRANSLITERATION_METADATA.file_configs.items():
  for src_script, trg_script in scripts:
    _ALL_SUB_TASK_FILES.append((
        f'transliteration/test/{src_script}2{trg_script}.{lang}.jsonl',
        f'{src_script}2{trg_script}.{lang}',
    ))
    _ALL_GOLD_DATA_FILES.append(
        f'transliteration/test/{src_script}2{trg_script}.{lang}.jsonl'
    )
  for trg_script, src_script in scripts:
    _ALL_SUB_TASK_FILES.append((
        f'transliteration/test/{src_script}2{trg_script}.{lang}.jsonl',
        f'{src_script}2{trg_script}.{lang}',
    ))
    _ALL_GOLD_DATA_FILES.append(
        f'transliteration/test/{src_script}2{trg_script}.{lang}.jsonl'
    )

# MT
for lang in _MT_METADATA.file_configs:
  _ALL_SUB_TASK_FILES.append((f'translation/test/en2{lang}.jsonl', lang))
  _ALL_GOLD_DATA_FILES.append(f'translation/test/en2{lang}.jsonl')

# NER
for lang in _NER_METADATA.file_configs:
  _ALL_SUB_TASK_FILES.append((f'ner/test/{lang}.jsonl', lang))
  _ALL_GOLD_DATA_FILES.append(f'ner/test/{lang}.jsonl')

# Semantic parsing
for lang in _SEMANTIC_PARSING_METADATA.file_configs:
  # Remove the locale information here so that the language code
  # is consistent with other tasks.
  if lang == 'pt_br':
    simple_lang = 'pt'
  else:
    simple_lang = lang
  _ALL_SUB_TASK_FILES.append(
      (f'semantic_parsing/test/{lang}.jsonl', simple_lang)
  )
  _ALL_GOLD_DATA_FILES.append(
      f'semantic_parsing/test/{lang}.jsonl'
  )

# QA-IN-LANG
for lang in _QA_IN_LANG_METADATA.file_configs:
  _ALL_SUB_TASK_FILES.append((f'qa_in_lang/test/{lang}.jsonl', lang))
  _ALL_GOLD_DATA_FILES.append(f'qa_in_lang/test/{lang}.jsonl')

# QA-CROSS-LANG
for lang in _QA_CROSS_LANG_METADATA.file_configs:
  _ALL_SUB_TASK_FILES.append((f'qa_cross_lang/test/{lang}.jsonl', lang))
  _ALL_GOLD_DATA_FILES.append(f'qa_cross_lang/test/{lang}.jsonl')

# Retrieval in-lang
for lang in _RETRIEVAL_IN_LANG_METADATA.file_configs:
  _ALL_SUB_TASK_FILES.append((f'retrieval_in_lang/test/{lang}.jsonl', lang))
  _ALL_GOLD_DATA_FILES.append(f'retrieval_in_lang/test/{lang}.jsonl')

# Retrieval cross-lang
for lang in _RETRIEVAL_CROSS_LANG_METADATA.file_configs:
  _ALL_SUB_TASK_FILES.append((f'retrieval_cross_lang/test/{lang}.jsonl', lang))
  _ALL_GOLD_DATA_FILES.append(f'retrieval_cross_lang/test/{lang}.jsonl')

# ASR
for lang in _ASR_METADATA.file_configs:
  # Remove the locale information for ASR lang codes so that they
  # are consistent with other tasks when calculating results
  # aggregated by language.
  # We differentiate the Mandarin from two locations by keeping
  # the indicating whether its traditional(hant) or
  # simplified (hans).
  if lang == 'cmn_hant_hk':
    simple_lang = 'cmn_hant'
  elif lang == 'cmn_hans_cn':
    simple_lang = 'cmn_hans'
  else:
    simple_lang = lang.split('_')[0]
  _ALL_SUB_TASK_FILES.append((f'asr/test/{lang}.jsonl', simple_lang))
  _ALL_GOLD_DATA_FILES.append(f'asr/test/{lang}.jsonl')


# Auto Complete
for lang in _AUTOCOMPLETE_METADATA.file_configs:
  _ALL_SUB_TASK_FILES.append((f'autocomplete/test/{lang}.jsonl', lang))
  _ALL_GOLD_DATA_FILES.append(f'autocomplete/test/{lang}.jsonl')


def _get_metadata(task_name: str) -> Optional[TaskMetadata]:
  for meta in _ALL_METADATA:
    if meta.match(task_name):
      return meta
  return None


def get_all_xtremeup_tasks() -> dict[str, list[Task]]:
  """Gets all registered XTREME-UP tasks, but with associated metadata (as `Task`)."""
  result = {}
  for (prediction_file, language), gold_file in zip(
      _ALL_SUB_TASK_FILES, _ALL_GOLD_DATA_FILES, strict=True
  ):
    meta: Optional[TaskMetadata] = _get_metadata(prediction_file)
    if meta is None:
      logging.info("No task metadata found for '%s'", prediction_file)
      continue
    task = Task(
        full_name=meta.simple_name,
        simple_name=meta.simple_name,
        lang=language,
        metric=meta.metric,
        output_file_name=prediction_file,
        gold_file_name=gold_file,
    )
    if task.simple_name not in result:
      result[task.simple_name] = []
    result[task.simple_name].append(task)
  return result


def average(values: Sequence[Optional[float]]) -> Optional[float]:
  # If any scores are `None`, the result is `None`.
  # We use this to indicate an incomplete evaluation.
  if any(value is None for value in values):
    return None
  return sum(values) / len(values) if values else 0.0


@dataclasses.dataclass
class SingleResult:
  """A single narrow task on a specific language within a task."""

  name: str  # Full name of the output file.
  metric: Optional[str] = None
  score: Optional[float] = None

  def set(
      self, metric: str, score: Optional[float], force_update: bool = False
  ) -> None:
    if self.score is not None and not force_update:
      raise ValueError(f"`score` already set for '{self.name}'.")
    self.score = score
    self.metric = metric


@dataclasses.dataclass
class LangResult:
  """A set of results for a language in the XTREME-UP benchmark."""

  name: str  # e.g. 'de'
  is_in_xtreme_up: bool
  results: dict[str, SingleResult] = dataclasses.field(default_factory=dict)

  @property
  def score(self) -> Optional[float]:
    """Returns the average score over all languages for this task."""
    return average([result.score for result in self.results.values()])

  @property
  def metric(self) -> str:
    """Returns the metric used for this task."""
    single_result = list(self.results.values())[0]
    result = single_result.metric
    if not result:
      raise ValueError(f'No metric found for {single_result.name}')
    return result

  def get(self, full_name: str) -> SingleResult:
    if full_name not in self.results:
      self.results[full_name] = SingleResult(full_name)
    return self.results[full_name]


@dataclasses.dataclass
class TaskResult:
  """A set of results for a task in the XTREME-UP benchmark."""

  name: str  # e.g. 'Transliteration'
  results: dict[str, LangResult] = dataclasses.field(default_factory=dict)
  _metric: Optional[str] = None

  @property
  def score(self) -> Optional[float]:
    """Returns the average score over all languages for this task."""
    return average(
        [
            result.score
            for result in self.results.values()
            if result.is_in_xtreme_up
        ]
    )

  @property
  def metric(self) -> str:
    """Returns the metric used for this task."""
    if not self._metric:
      self._metric = list(self.results.values())[0].metric
    return self._metric

  def get(self, full_name: str) -> LangResult:
    lang_code = full_name
    # For transliteration, convert "Latn2Ethi.am" to "am" before checking
    # against the list of under-represented languages.
    if "." in lang_code:
      lang_code = lang_code.split(".")[1]
    if full_name not in self.results:
      self.results[full_name] = LangResult(
          full_name,
          is_in_xtreme_up=(
              lang_code in constants.XTREME_UP_UNDER_REPRESENTED_LANGS
          ),
      )
    return self.results[full_name]


@dataclasses.dataclass
class Results:
  """All results for a complete submission."""

  submission_name: str
  results: dict[str, TaskResult] = dataclasses.field(default_factory=dict)

  @property
  def score(self) -> Optional[float]:
    """Returns an aggregated utility score for this run."""
    scores = []
    for task in self.results.values():
      # We modify the metrics that are better with a lower value so that all the
      # scores go in the aggregated final results are better with higher values.
      if task.metric.lower() == 'cer' and task.score is not None:
        scores.append(100 - task.score)
      elif task.metric.lower() == 'mrr' and task.score is not None:
        scores.append(100 * task.score)
      else:
        scores.append(task.score)
    return average(scores)

  def get(self, name: str) -> TaskResult:
    if name not in self.results:
      self.results[name] = TaskResult(name)
    return self.results[name]


def score_submission(
    submission_name: str,
    submission_folder: str,
    gold_data_folder: str,
    results: Optional[Results],
    update_task_name: Optional[str],
    instantiate_results_object: Optional[bool] = False,
) -> Results:
  """Calculates the scores for a submission and output the results."""
  xtremeup_tasks: dict[str, list[Task]] = get_all_xtremeup_tasks()
  logging.info('Found %d xtreme-up tasks.', len(xtremeup_tasks))
  # Creates a new results object if we are not modifying an existing one.
  if not results:
    results = Results(submission_name=submission_name)
  if update_task_name is not None:
    if update_task_name not in xtremeup_tasks:
      raise ValueError(f"Unrecognized task name: '{update_task_name}'")
    task = xtremeup_tasks[update_task_name]
    tasks_to_score = [task]
    force_update_score = True
  else:
    tasks_to_score = list(xtremeup_tasks.values())
    force_update_score = False
  for xtremeup_task_for_all_langs in tasks_to_score:
    for xtremeup_task in xtremeup_task_for_all_langs:
      task_result: TaskResult = results.get(xtremeup_task.simple_name)
      lang_task_result: LangResult = task_result.get(xtremeup_task.lang)
      single_task_result: SingleResult = lang_task_result.get(
          xtremeup_task.full_name
      )

      task_output_file = os.path.join(
          submission_folder, xtremeup_task.output_file_name
      )
      gold_output_file = os.path.join(
          gold_data_folder, xtremeup_task.gold_file_name
      )
      if instantiate_results_object:
        single_task_result.set(
            metric=xtremeup_task.metric,
            score=None,
            force_update=True
        )
        continue
      score = metrics.score_file(
          gold_jsonl_file=gold_output_file,
          prediction_jsonl_file=task_output_file,
          metric_name=xtremeup_task.metric,
      )

      logging.info(
          'Task %s subtask %s language %s with metric %s scored: %.1f',
          xtremeup_task.simple_name,
          xtremeup_task.full_name,
          xtremeup_task.lang,
          xtremeup_task.metric,
          score,
      )
      single_task_result.set(
          metric=xtremeup_task.metric,
          score=score,
          force_update=force_update_score,
      )
  return results


def write_jsonl_results(results: Results, path: str) -> dict[str, Any]:
  """Dumps the evaluation results to a jsonl file."""
  jsonl_data = {}
  jsonl_data['submission_name'] = results.submission_name
  jsonl_data['score'] = results.score
  jsonl_data['tasks'] = {}
  jsonl_data['languages'] = {}

  for task_name, task_results in results.results.items():
    jsonl_data['tasks'][task_name] = {
        'score': task_results.score,
        'metric': task_results.metric,
        'languages': {},
    }
    for lang_name, lang_result in task_results.results.items():
      jsonl_data['tasks'][task_name]['languages'][lang_name] = {
          'score': lang_result.score,
          'metric': lang_result.metric,
          'results': {},
          'is_in_xtreme_up': lang_result.is_in_xtreme_up,
      }
      # We also keep the score of each individual output for the task.
      for result in lang_result.results.values():
        jsonl_data['tasks'][task_name]['languages'][lang_name]['results'][
            result.name
        ] = result.score
      # Update the language level results.
      if lang_name not in jsonl_data['languages']:
        jsonl_data['languages'][lang_name] = {
            'score': [],
            'is_in_xtreme_up': lang_result.is_in_xtreme_up,
        }
      if (
          lang_result.metric.lower() == 'cer'
          or lang_result.metric.lower() == 'cer_normalized'
      ):
        if lang_result.score is not None:
          jsonl_data['languages'][lang_name]['score'].append(
              100.0 - lang_result.score
          )
      else:
        jsonl_data['languages'][lang_name]['score'].append(lang_result.score)

  for lang_name, lang_info in jsonl_data['languages'].items():
    jsonl_data['languages'][lang_name]['score'] = average(lang_info['score'])

  with file_utils.open(path, 'w') as f:
    json.dump(jsonl_data, f, indent=2)
  return jsonl_data


def get_results_from_json(
    result_json_file: str, results: Results
) -> Results:
  """Reads in the existing results.json."""
  # Reads in the results for the given submissions.
  result_json = json.load(file_utils.open(result_json_file, 'r'))
  for task_name, task_result_json in result_json['tasks'].items():
    task_result: TaskResult = results.get(task_name)
    for language, lang_result_json in task_result_json['languages'].items():
      lang_task_result: LangResult = task_result.get(language)
      metric = lang_result_json['metric']
      for full_name, score in lang_result_json['results'].items():
        single_task_result: SingleResult = lang_task_result.get(full_name)
        logging.info(
            (
                'Reads in existing results: Task %s Subtask %s with metric %s'
            ),
            task_name,
            full_name,
            metric,
        )
        single_task_result.set(metric=metric, score=score)
  return results


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Instantiate a results object with all tasks and languages to score.
  results = score_submission(
      _SUBMISSION_NAME.value,
      _SUBMISSION_FOLDER.value,
      _GOLD_DATA_FOLDER.value,
      None,
      update_task_name=None,
      instantiate_results_object=True
  )
  # Reads in existing results if it exists.
  result_json_file = file_utils.join(
      [_SUBMISSION_FOLDER.value, 'results.json']
  )
  if file_utils.exists(result_json_file):
    results = get_results_from_json(
        result_json_file, results
    )
  # Actually scoring the tasks
  results = score_submission(
      _SUBMISSION_NAME.value,
      _SUBMISSION_FOLDER.value,
      _GOLD_DATA_FOLDER.value,
      results,
      update_task_name=_SCORE_TASK_NAME.value,
      instantiate_results_object=False
  )
  # Write the result of this submission to a json file.
  _ = write_jsonl_results(results, _JSON_OUT.value)

if __name__ == '__main__':
  app.run(main)
