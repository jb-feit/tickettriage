-- Runs once, when the database volume is first created.
CREATE EXTENSION IF NOT EXISTS vector;

-- ops: operational tables written by the api (predictions, vectors, labels).
-- dw:  warehouse star schema loaded by the ETL job.
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS dw;

-- Tables are added in weeks 2-3.
