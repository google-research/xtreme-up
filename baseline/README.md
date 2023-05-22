# XTREME-UP Baseline Systems

We provide simple mT5 and ByT5 baseline systems as a starting point for
XTREME-UP.

**Preprocessing:** The pre-processing script in this directory
`jsonl_to_seqio_tsv.py` and converts the structured JSONL inputs into linear
`inputs` and `targets`; it does not use any framework-specific code and should
be useful to users of any framework. However, the pre-processing code is
intentionally left separated from the original inputs to allow for future
innovation.

**Modeling:** The core modeling code uses T5X (T5 models implemented in Jax)
with seqio for pre-processing and post-processing.

**Evaluation:** Similarly, the evaluation of text output is separately factored
out from the modeling code and is not framework-dependent. See the
[evaluation/](../evaluation) directory for details.

## Running the baseline systems

### Preprocessing

```sh
python3 jsonl_to_seqio_tsv.py \
  --input_dir=${JSONL_DIR_IN} \
  --output_dir=${TSV_DIR_OUT} \
  --task=mt
```

### Fine-tuning

Consider reading [https://github.com/google-research/t5x] as a quick start guide
to T5X. It can be run either locally or on Google Cloud using the `xm_launch.py`
script.

Below is an example of the flags you might provide to T5X. You will almost
certainly need to modify these based on what environment you choose to run in
(e.g. using Google Cloud vs locally):

```sh
python3 ${T5X_DIR}/t5x/train.py \
  --model_dir=${MODEL_DIR} \
  --gin.TASK_TSV_DATA_DIR=\${TSV_DATA_DIR}/qa_in_lang'\' \
  --gin_search_paths=baseline \
  --gin_file=byt5/base_finetune.gin \
  --gin_file=tasks_lib.gin \
  --gin.MIXTURE_OR_TASK_NAME=\'xtreme_up_qa_in_lang_mt5\' \
  --gin.MIXTURE_OR_TASK_MODULE=\'xtreme_up.baseline.qa_tasks\' \
  --gin.USE_CACHED_TASKS=False \
  --gin.BATCH_SIZE=64 \
  --gin.TASK_FEATURE_LENGTHS=\{\'inputs\':\ 1024,\ \'targets\':\ 128\} \
  --gin.TRAIN_STEPS=1003580 \
  --gin.EVAL_PERIOD=100000 \
  --gin.JSON_WRITE_N_RESULTS=20 \
  --gin.train.train_eval_dataset_cfg=None \
  --gin.train.infer_eval_dataset_cfg=None \
  --gin.utils.SaveCheckpointConfig.period=100000 \
  --gin.utils.DatasetConfig.pack=False
```

In the baseline systems, each task is fine-tuned individually.

### Inference

Note that the example below shows inference on the test set; however, we
strongly recommend iterating on the validation set and measuring on the test set
infrequently.

```sh
python3 ${T5X_DIR}/t5x/infer.py \
  --gin_file=byt5/base_eval.gin \
  --gin_search_paths=baseline \
  --output_base_dir=$HOME/xtreme_up_out
  --gin_file=tasks_lib.gin \
  --gin.TASK_TSV_DATA_DIR=\'${TSV_DATA_DIR}/qa_in_lang\' \
  --gin.EVAL_OUTPUT_DIR=\'${INFER_DIR}' \
  --gin.CHECKPOINT_PATH=\'${MODEL_DIR}/checkpoint_1003580\' \
  --gin.MIXTURE_OR_TASK_NAME=\'xtreme_up_qa_in_lang_mt5\' \
  --gin.MIXTURE_OR_TASK_MODULE=\'xtreme_up.baseline.qa_tasks\' \
  --gin.utils.DatasetConfig.split=\'test\'
```

This will produce output that can then be run through the evaluation framework
(see above).

## Task Notes

### ASR

We first run the MAESTRO-U system on the audio files included with XTREME-UP.
mT5 or ByT5 can then be optionally added as a second stage. We expect
multi-modal systems to soon unify these stages.

### OCR

We use Google OCR as our baseline system. As with ASR, we expect to see
improvements as multimodal models unify these paradigms.

### Retrieval

Retrieval inference will produce vectors for questions and passages; the
evaluation framework includes a tiny framework-agnostic nearest neighbors script
that will make passage ID predictions.

### Autocomplete

Unlike the other text tasks, autocomplete will produce top-3 output (see
`top3.gin`).
