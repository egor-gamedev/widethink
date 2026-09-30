# 8. A dependency-free hashing embedder as the default

Date: 2026-09-30 · Status: Accepted

## Context

Satiation, inhibition of return and reinstatement need a similarity between
ideas. Neural embedders need either a network service (and a key, which Claude
users may not have) or a heavy local model. The library should run, and its
tests should pass, with no service and no model download.

## Decision

- The default embedder hashes words, word pairs and character trigrams with
  BLAKE2b into a 4096-dimensional vector (Python's `hash` is salted per process
  and would break reproducibility). Weights and thresholds were calibrated on
  labelled pairs in English and Russian: duplicates ≥ 0.84, paraphrases
  0.55–0.79, unrelated ≤ 0.04; thresholds 0.2 (related) and 0.8 (duplicate).
- Every embedder declares its own thresholds, because cosine values are not
  comparable across models.
- Documentation recommends a neural embedder (OpenAI-compatible or
  sentence-transformers) for real runs.

## Consequences

- Offline demos and tests are fast and deterministic.
- The default sees surface form, not meaning: rephrasings with different words
  are missed. Real experiments must use and report a neural embedder.
