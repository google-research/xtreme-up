# Evaluation

## Step 1: Convert output format

First, you'll need to convert your outputs from your system's output format. We
provide some helper scripts to do this (see
`move_outputs_to_example_submission.py`). For our baseline system, you would
run: `move_outputs_to_example_submission.sh`, which calls the python script:


```
python move_outputs_to_example_submission.py
  --task=TASK_NAME
  --system_output_dir=DIR_TO_ORIGINAL_OUTPUT
  --submission_dir=DIR_TO_XTREME_UP_SUBMISSION
  --checkpoint_step=SELECTED_CHECKPOINT_FOR_EVAL
```

Modify `move_outputs_to_example_submission.py` to read your own system's output
format, if needed. This will produce the JSONL predictions that you'll upload
with your XTREME-UP submission and write them to the submission folder you
specify.

## Step 2: Score the submission

Next, you'll score your outputs for the various tasks. It's okay if you haven't
run on all of the tasks, we'll evaluate partial results. Run `score_system.sh`
on your submission folder to do this. This script calls `score_system.py`, which
can be used as:


```
python score_system.py
  --submission_name=NAME_OF_YOUR_SUBMISSION
  --submission_folder=DIR_TO_YOUR_XTREME_UP_SUBMISSION
  --gold_data_folder=DIR_TO_REFERENCES
  --json_out=DIR_TO_YOUR_XTREME_UP_SUBMISSION/results.json
```

Then you'll find a file `results.json` in that
directory.

## Step 3: Add your results to the public result tracker

Finally, if you have a cool result you'd like to share with the research
community, run `update_public_results.sh`, which will regenerate
FULL_RESULTS.md to include your system.

## Step 4: Send us a pull request!

Now just send a GitHub Pull Request to
our repo that includes your predicions.jsonl files, your results.json file, and
the updated FULL_RESULTS.md!
