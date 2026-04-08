# Submission Guidelines — Challenge-Specific (Python Code Golf Fine-Tuning)

These rules are specific to this challenge and repository workflow.

## 1) Output Contract

- Graded file path: **`./working/submission.csv`**.
- CSV header must be exactly: `id,code`.
- Must contain exactly **200** rows for test IDs **1..200**.
- Each ID must appear **once** (no missing, no duplicates).
- `code` must be non-empty and a **single Python lambda expression**.

## 2) Grader Behavior

A row receives 0 if any of these occur:

- parse failure,
- runtime error,
- timeout,
- unsafe return type,
- output mismatch on any visible/hidden example.

Difficulty-weighted scoring uses:

$$
\text{score} = 100 \cdot \frac{\sum_i \text{difficulty}_i \cdot \left(\frac{|\text{reference}_i|}{|\text{submitted}_i|}\right)}{\sum_i \text{difficulty}_i}
$$

Correctness is mandatory; shorter correct lambdas improve score.

## 3) Allowed Runtime Surface for Generated Code

### Allowed standard-library imports (via `__import__`)

- `itertools`, `math`, `re`, `functools`, `statistics`, `collections`, `string`

### Allowed builtins

`abs`, `all`, `any`, `bool`, `bytes`, `chr`, `dict`, `divmod`, `enumerate`, `filter`, `float`, `frozenset`, `hash`, `hex`, `int`, `iter`, `len`, `list`, `map`, `max`, `min`, `next`, `oct`, `ord`, `pow`, `range`, `repr`, `reversed`, `round`, `set`, `slice`, `sorted`, `str`, `sum`, `tuple`, `type`, `zip`, `__import__`.

## 4) Repository Workflow (Recommended)

From project root:

1. Build submission artifact:
   - `python solution.py`
   - or `./scripts/make_submission.sh`
2. Validate CSV shape/format:
   - `python -m golf_ft.submission_qa --csv working/submission.csv`
3. Optional visible replay check:
   - `python -m golf_ft.submission_qa --csv working/submission.csv --replay-visible`

## 5) Challenge-Specific Final Checklist

- [ ] `solution.py` runs end-to-end in a clean session.
- [ ] Output is written to `./working/submission.csv`.
- [ ] CSV has header `id,code`.
- [ ] CSV includes all IDs 1..200 exactly once.
- [ ] Every `code` entry is a valid single lambda and non-empty.

## 6) Challenge-Specific Pitfalls

- Writing to wrong path (must be `./working/submission.csv`).
- Emitting non-lambda code in the `code` column.
- Duplicate/missing IDs.
- Assuming visible-only checks guarantee hidden correctness.
- Approaches violating challenge intent (manual per-row solving or benchmark exploitation).

## 7) Definition of Done

A clean run generates `./working/submission.csv` that passes QA and contains valid lambda predictions for all 200 test IDs.
