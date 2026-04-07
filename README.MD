Overview
The goal of this challenge is to fine-tune a pretrained LLM that will find shortest Python lambda solutions for small algorithmic tasks.

Each sample provides a natural-language specification in description, a set of input/output examples in examples, and a numeric difficulty score.

The dataset mixes several task styles, including short arithmetic, comparison, boolean, indexing, list, string, grouping, parsing, recursion, dynamic-programming, and pattern-matching tasks. Reference solutions are compact one-line lambdas, and the metric rewards both correctness and brevity, so shorter correct code scores better than longer correct code.

The intended way to solve this benchmark is to fine-tune a pretrained LLM to generate short Python lambdas from the task text and examples. Submissions that primarily exploit the dataset, grader, hidden examples or sandbox rather than solve the tasks as intended will be rejected.

The following submission behaviors are not allowed:

manually finding solution for each sample
relying on blocked libraries or blocked builtins inside submitted code
using reflection, interpreter internals or any bypass tricks instead of solving the tasks as intended
any method intended primarily to exploit the benchmark, hidden examples, grader behavior or dataset structure rather than solve the tasks themselves
using LLM API
Any submission found to rely on these disallowed approaches will be rejected even if it achieves a high score.

To sum up: you have to fine-tune a pretrained LLM, any other approach will be rejected.

Evaluation
On the challenge platform, scoring is “higher is better” with an unbounded upper range (in practice the write-up cites an effective maximum around 120 on posted tasks).

Submissions are evaluated using difficulty-weighted score.

For each test row:

the submitted code string is parsed and evaluated inside a fresh isolated worker process
the submitted code must parse to a single lambda expression
that lambda is executed on every visible and hidden example for that row
if parsing fails, execution raises an error, execution times out, the return type is unsafe, or any returned value differs from the expected example["output"], that row receives 0
For correct rows, the grader computes:


golf_score = len(reference_code.strip()) / len(submitted_code.strip())

score = 100 * sum(difficulty_i * golf_score_i) / sum(difficulty_i)

where:

reference_code is the hidden gold solution for that test row
submitted_code is your submitted code string
difficulty_i is the difficulty value for that row
This means:

incorrect rows contribute 0
correct but longer solutions score below the reference baseline for that row
correct solutions matching the reference length score 1.0 for that row before weighting
shorter correct solutions score above 1.0 for that row before weighting
harder tasks contribute more because of the difficulty weight
The grader applies a 5 second timeout per test row, uses a memory cap and compares outputs type-sensitively. For example, True does not count as 1.

Scores are therefore interpreted as follows:

0.0 means nothing was solved correctly
100.0 means every test task was solved correctly at exactly the hidden reference-code length
above 100.0 is possible if a submission beats the reference solutions on code length
The true shortest solutions are not known, so there is no strict hard cap. The effective maximum is estimated around 120, while matching all reference solutions gives score 100.

Dataset
The source dataset contains 240 Python function-synthesis tasks in total.

train.jsonl: 40 rows
test.jsonl: 200 rows
Each task is self-contained and includes examples that define the intended behavior of the target function.

For every test row, exactly 4 examples are hidden for private evaluation and the remaining examples are released in test.jsonl.

Answers are heterogeneous and may be JSON-serializable Python values such as:

integers
strings
booleans
lists
dictionaries
floating-point values
Difficulty scores are floating-point values on an approximate 1.0 to 10.0 scale, where larger values indicate harder tasks.

File descriptions
train.jsonl - training set with fields: id, description, code, examples, difficulty
test.jsonl - test set with fields: id, description, examples, difficulty
sample_submission.csv - example submission file with columns: id, code
Column descriptions
id - integer identifier, unique within the released split
description - natural-language specification of the target function
code - reference solution for train rows; prediction target for test rows
examples - list of dictionaries with input and output keys
difficulty - floating-point difficulty score included in both train and test
Each examples entry has the form:


{"input": [1, 2], "output": 3}

The released split uses dense integer ids starting at 1 within each split.

Allowed Python subset
The official grader evaluates submissions inside an isolated worker that accepts only a small expression-oriented subset of Python. Submissions must be single lambda expressions.

Allowed standard-library modules
The only standard-library modules that may be imported, via __import__, are:

itertools
math
re
functools
statistics
collections
string
Allowed builtins
The grader exposes only this builtin surface:

abs
all
any
bool
bytes
chr
dict
divmod
enumerate
filter
float
frozenset
hash
hex
int
iter
len
list
map
max
min
next
oct
ord
pow
range
repr
reversed
round
set
slice
sorted
str
sum
tuple
type
zip
__import__ (only for the allowed modules above)
Submission
You must submit a submission.csv file with a header and one row per test task.

id: test task id
code: predicted single Python lambda expression
Example:


id,code

1,"lambda a,b:a+b"

2,"lambda x:-x"

3,"lambda s:s[::-1]"

Requirements
The test set contains 200 rows with ids 1..200, so the submission file must contain exactly 200 predictions
Must include the header row: id,code
Must include each id exactly once, with no missing ids and no duplicates
Must not contain missing or empty values in the code column
Must save predictions to submission.csv

Official platform submission (Project Eris)

When you submit on the challenge site, the grader expects your run to write predictions to this relative path:

- **Output file:** `./working/submission.csv`

The released **dataset (.zip)** is laid out like this:

- `dataset/public/sample_submission.csv`
- `dataset/public/test.jsonl`
- `dataset/public/train.jsonl`

Your submission package typically includes:

1. **solution.py** or a **Jupyter notebook (.ipynb)** (required) — your training and inference code.
2. **submission.csv** (required) — the predictions file (same schema as above; must match the platform path if their runner expects `./working/submission.csv`).
3. **Time spent** (required) — total hours for the attempt (training, iteration, research).
4. **Your approach** (required) — short description: model, preprocessing, what worked or failed.

The site also tracks **submission credits** (a limited number of graded submissions per challenge); details are on the problem page.

Submissions are **graded against the current problem version** on the site (when the problem is updated, newer versions replace older ones for scoring).

## Local workflow (this repo)
**Remote Linux GPU (recommended for training):** step-by-step guide in [LINUX_GPU_RUNBOOK.md](LINUX_GPU_RUNBOOK.md) (CUDA venv, `working/submission.csv`, scp, troubleshooting).

On your own machine: `./scripts/setup_venv.sh`, then `source .venv/bin/activate`.

**Hugging Face Hub mirror:** If `huggingface.co` is slow or blocked, point the Hub client at [hf-mirror.com](https://hf-mirror.com) before any command that downloads models:

```bash
source scripts/env_hf_mirror.sh
```

This sets `HF_ENDPOINT=https://hf-mirror.com` (used by `transformers` / `huggingface_hub`). You can also run `export HF_ENDPOINT=https://hf-mirror.com` once per shell.

Build training JSON: `python -m golf_ft.data_pipeline`.
Fine-tune: `python -m golf_ft.train_lora` (optional `--max-steps N` for a smoke run).
One-shot pipeline (train + infer + QA): `./scripts/run_pipeline.sh` — set `SMOKE=1` for a short run (`--max-steps 2` and `--model-row-limit 5` on infer); set `NO_MIRROR=1` to skip `hf-mirror.com`.
Infer: `python -m golf_ft.infer` (writes `submission.csv` at the repo root by default and runs shape QA unless `--skip-qa`; uses several generation attempts by default and keeps the shortest visible-correct lambda — add `--quick-infer` for only two attempts). For a cheap connectivity test use `--model-row-limit N` (first N tasks use the model, rest `lambda x:x`; use `-1` / omit for all 200). For the platform path above, use `python -m golf_ft.infer --out working/submission.csv` after creating a `working` directory (or copy the CSV there before upload).
Validate a CSV only: `python -m golf_ft.submission_qa --csv submission.csv` (add `--replay-visible --test test.jsonl` to re-check visible examples).
Train-set weighted score (references from `train.jsonl`): `python -m golf_ft.score_train --csv path/to.csv` — CSV must list **train** task ids (e.g. a holdout prediction file), not the 200-row test submission.
