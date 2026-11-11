# Week 1 plan: foundations

Goal by Friday: **a baseline accuracy number, a container that reproduces it, and a clean
repo with green CI.** No AWS spending beyond a few cents is needed this week.

## Checklist

### Day 1: repo and environment
- [ ] Create a GitHub repo, push this folder (`git init`, first commit, push to `main`).
- [ ] Create the virtualenv and install requirements (see README quickstart).
- [ ] `make test` passes locally; the GitHub Actions run on your first push is green.

**Done when:** CI shows a green check on `main`.

### Day 2: AWS account and guardrails (see [AWS_SETUP.md](AWS_SETUP.md))
- [ ] Account created, root user has MFA, you work as an IAM/SSO user.
- [ ] **Budget alert** created before anything else.
- [ ] Region chosen and written into `.env`.

**Done when:** you can run `aws sts get-caller-identity` from your terminal.

### Day 3: dataset and split
- [ ] `make data` runs; read the printed launch/future intent lists.
- [ ] Open `data/processed/meta.json` and note the row counts.
- [ ] Skim 20 random rows per split. Do the tickets look sensible?

**Done when:** `data/processed/` has five parquet files and `meta.json`.

### Day 4: embeddings and baseline
- [ ] `make embed` (local model is fine) then `make train`.
- [ ] Read `artifacts/baseline/*/metrics.json` and `report.txt`.
- [ ] Write down: val/test accuracy, macro-F1, auto-route rate at 0.7, and the
      `unseen_intent_probe` confidence compared with the test confidence.

**Done when:** you have the baseline numbers in your notes.

### Day 5: Docker
- [ ] `make docker-train` reproduces the same numbers inside a container.
- [ ] `make db-up`, then `make db-psql` and run `\dx` to confirm the `vector` extension.

**Done when:** the container run matches your local metrics, and pgvector is installed.

### Day 6: Bedrock and AWS resources (can slip to the weekend)
- [ ] Enable model access in the Bedrock console (Titan Text Embeddings V2, plus a Claude
      model if offered in your region).
- [ ] `make check-bedrock` prints an embedding; with `BEDROCK_JUDGE_MODEL_ID` set,
      `python -m tickettriage.check_bedrock --judge` prints `OK`.
- [ ] Create the S3 bucket, ECR repository and EC2 instance role. **Do not launch the EC2
      instance yet** (week 2) so it doesn't bill while idle.

**Done when:** both Bedrock checks succeed and the three AWS resources exist.

## Decide this week (write the answers in the decision log below)

1. Embedding model for the whole project: local MiniLM or Bedrock Titan V2? Changing later
   means re-embedding everything.
2. Confidence threshold for "auto-route" (the starting point is 0.7).
3. Which dashboard tool: Grafana or Streamlit?

## Notes for the report

- Dataset facts: size, intents, license, how duplicates were handled.
- Why intents are split launch/future, and why `test` is frozen.
- Baseline numbers and a sentence on why they are high (templated data).
- The Docker image: base image, size, and what is mounted from the host.

## Decision log

| Date | Decision | Why |
|---|---|---|
| | Embedding model: | |
| | Auto-route threshold: | |
| | Dashboard tool: | |

## If you fall behind

Skip Day 6 (Bedrock and AWS resources) and do it at the start of week 2. The baseline,
Docker image and CI are the parts later weeks depend on.
