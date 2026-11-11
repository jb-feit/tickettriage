# Run these inside your activated virtualenv (see README quickstart).
PY ?= python

.PHONY: data embed train all test lint check-bedrock docker-build docker-train db-up db-down db-psql

data:            ## download + clean + split the dataset
	$(PY) -m tickettriage.data

embed:           ## embed every split with the backend chosen in .env
	$(PY) -m tickettriage.embeddings

train:           ## train and evaluate the baseline classifier
	$(PY) -m tickettriage.train

all: data embed train

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

check-bedrock:   ## smoke-test Bedrock access (embeddings; add --judge for an LLM)
	$(PY) -m tickettriage.check_bedrock

docker-build:
	docker build -f docker/Dockerfile.train -t tickettriage-train:dev .

# Mounts data/ and artifacts/ so results land on your machine.
# Mounts ~/.aws read-only so the container can reach Bedrock if EMBEDDING_BACKEND=bedrock.
docker-train: docker-build
	docker run --rm --env-file .env \
	  -v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/artifacts:/app/artifacts" \
	  -v "$(HOME)/.aws:/root/.aws:ro" \
	  tickettriage-train:dev

db-up:
	docker compose up -d db

db-down:
	docker compose down

db-psql:
	docker compose exec db psql -U $${POSTGRES_USER:-tickettriage} -d $${POSTGRES_DB:-tickettriage}
