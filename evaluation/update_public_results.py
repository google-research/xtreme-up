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
import json
from typing import Any, Optional

from absl import app
from absl import flags
import prettytable

from xtreme_up.evaluation import file_utils


_MARKDOWN_OUT = flags.DEFINE_string(
    "markdown_out",
    None,
    (
        "Output path to main markdown report file."
    ),
    required=False,
)

_SUBMISSION_BASE_DIR = flags.DEFINE_string(
    "submission_base_dir",
    None,
    "The jsonl file containing the evaluation results for all_submissions.",
    required=True,
)


def _pretty_score(score: Optional[float], metric: Optional[str] = None) -> str:
  """Turns `score` into a pretty string with `metric` information.

  Args:
    score: The score value. `None` if it is unavailable or one of the child
      scores being aggregated was unavailable.
    metric: The metric (units) of `score`, if available.

  Returns:
    A pretty stringified score.
  """

  def _format(score: float, metric: Optional[str]) -> str:
    # Only show one digit of precision, since more is typically meaningless.
    if metric:
      return f"{score:.1f} ({metric})"
    else:
      return f"{score:.1f}"

  if score is None:
    # This happens when we encounter a task with no score.
    return "--"
  return _format(score, metric)


def get_full_results_table(
    full_results_jsonl: list[dict[str, Any]]
) -> prettytable.PrettyTable:
  """Dumps individual task-level results across all areas/tasks."""
  table = prettytable.PrettyTable()
  table_field_names = ["Submission", "Aggregated Score", "Ranking"]
  task_names = list(full_results_jsonl[0]["tasks"].keys())
  for task_name in task_names:
    metric = full_results_jsonl[0]["tasks"][task_name]["metric"]
    table_field_names.append(f"{task_name} ({metric})")
  table.field_names = table_field_names

  ranking = 1
  for result_json in full_results_jsonl:
    row = [
        result_json["submission_name"],
        _pretty_score(result_json["score"]),
        str(ranking),
    ]
    for task_name, task_info in result_json["tasks"].items():
      row.append(_pretty_score(task_info["score"]))
    table.add_row(row)
    ranking += 1
  return table


def get_language_breakdown_table(
    full_results_jsonl: list[dict[str, Any]],
    xtreme_up_langs: list[str],
    other_langs: list[str],
) -> prettytable.PrettyTable:
  """Dumps individual task-level results across all areas/tasks."""
  table = prettytable.PrettyTable()
  table_field_names = ["Submission", "Aggregated Score", "Ranking"]
  table_field_names = table_field_names + xtreme_up_langs
  for lang in other_langs:
    table_field_names.append(f"({lang})")

  table.field_names = table_field_names
  ranking = 1
  for result_json in full_results_jsonl:
    row = [
        result_json["submission_name"],
        _pretty_score(result_json["score"]),
        str(ranking),
    ]
    for lang in xtreme_up_langs+other_langs:
      lang_info = result_json["languages"][lang]
      row.append(_pretty_score(lang_info["score"]))
    table.add_row(row)
    ranking += 1
  return table


def get_system_results_table(
    result_json: dict[str, Any],
    xtreme_up_langs: list[str],
    other_langs: list[str],
) -> prettytable.PrettyTable:
  """Dumps individual task-level results across all areas/tasks."""
  table = prettytable.PrettyTable()
  table_field_names = ["Task", "Aggregated Score"]

  table_field_names = table_field_names + xtreme_up_langs
  for lang in other_langs:
    table_field_names.append(f"({lang})")

  table.field_names = table_field_names
  for task, task_info in result_json["tasks"].items():
    metric = task_info["metric"]
    row = [f"{task} ({metric})", _pretty_score(task_info["score"])]
    for lang_name in xtreme_up_langs+other_langs:
      if lang_name in task_info["languages"]:
        row.append(_pretty_score(task_info["languages"][lang_name]["score"]))
      else:
        row.append("--")
    table.add_row(row)
  return table


def as_markdown(table: prettytable.PrettyTable) -> str:
  table.set_style(prettytable.MARKDOWN)
  result = str(table)
  table.set_style(prettytable.DEFAULT)
  return result


def to_markdown(full_results_jsonl: list[dict[str, Any]]) -> str:
  """Converts results to Markdown format."""

  # TODO(xinyiwang): add section about system info, such as name, affiliation,
  # Reproducible description of pre-training data, Reproducible description
  # of training/fine-tuning strategy, Hyperparameters, Notes, Closest Comparable
  # System, Differences from Comparable systems.
  result = f"""
## Overall Results

{as_markdown(get_full_results_table(full_results_jsonl))}

"""

  xtreme_up_langs = []
  other_langs = []
  for lang in full_results_jsonl[0]["languages"].keys():
    if full_results_jsonl[0]["languages"][lang]["is_in_xtreme_up"]:
      xtreme_up_langs.append(lang)
    else:
      other_langs.append(lang)

  result += f"""
## Overall language-wise breakdown

We also include the performance on higher resourced languages (in parenthesis) which are NOT included in the aggregated score of the XTREME-UP benchmark.

{as_markdown(get_language_breakdown_table(full_results_jsonl, xtreme_up_langs, other_langs))}

## Language-wise breakdown for each system

"""

  for result_json in full_results_jsonl:
    submission_name = result_json["submission_name"]
    result += f"""
### {submission_name}

{as_markdown(get_system_results_table(result_json, xtreme_up_langs, other_langs))}

"""
  return result


def write_markdown_results(
    full_results_data: list[dict[str, Any]], path: str
) -> None:
  markdown = to_markdown(full_results_data)
  with file_utils.open(path, "w") as f:
    f.write(markdown)


def get_all_results(submissions_base_dir: str) -> list[dict[str, Any]]:
  """Adds an entry to the overall results and re-order the submissions based on the average score."""
  # Reads in the results for all submissions.
  full_results_data = []
  for submission_dir in file_utils.listdir(submissions_base_dir):
    result_json_file = file_utils.join(
        [submissions_base_dir, submission_dir, "results.json"]
    )
    full_results_data.append(json.load(file_utils.open(result_json_file, "r")))

  # Sort the submissions based on the average score across tasks.
  def results_sort_fn(jsonl_data):
    return (jsonl_data["score"] is not None, jsonl_data["score"])

  full_results_data = sorted(
      full_results_data,
      key=results_sort_fn,
      reverse=True,
  )

  return full_results_data


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Read in results for all submissions.
  full_results_data = get_all_results(_SUBMISSION_BASE_DIR.value)
  # Write the full results jsonl to a markdown table.
  write_markdown_results(full_results_data, _MARKDOWN_OUT.value)


if __name__ == "__main__":
  app.run(main)
