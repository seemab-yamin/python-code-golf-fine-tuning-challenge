# Submission Guidelines — General (Platform-Level)

These are reusable platform-level requirements independent of any one challenge.

## 1) Core Requirements

- Submission must be **self-contained** and run end-to-end without manual intervention.
- Use **relative paths** from run root.
- Read input data from **`./dataset/public/`**.
- Write output artifacts only under **`./working/`**.
- Use only supported environment libraries (Kaggle Docker-style stack).
- Keep runtime within platform envelope (target: **under 30 minutes** on 64GB RAM + NVIDIA A10G 24GB VRAM).

## 2) Required Submission Artifacts

1. Entrypoint code (`solution.py` or `solution.ipynb`, depending on challenge rules).
2. Prediction artifact at the required output path.
3. Time spent (hours).
4. Short approach summary.

## 3) Submission Credits

- Max credits per problem: **6**.
- Refill rate: **1 credit every 4 hours**.
- Full refill time: **24 hours**.
- Credits are consumed only when a graded submit is made.

## 4) Universal Pre-Upload Checklist

- [ ] Clean run works end-to-end.
- [ ] Paths are relative and restricted to expected data/output folders.
- [ ] Output file exists at required location.
- [ ] Submission metadata (time spent + approach) is ready.
- [ ] Runtime is within platform limits.

## 5) Universal Failure Modes

- Hardcoded absolute paths.
- Output written to wrong directory.
- Manual steps required to finish run.
- Unsupported dependencies.
- Runtime exceeding platform limit.
