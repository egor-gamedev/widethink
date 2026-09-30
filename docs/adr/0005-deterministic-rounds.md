# 5. Deterministic rounds for parallel streams

Date: 2026-09-30 · Status: Accepted

## Context

Parallel streams should think concurrently (separate API calls) and see each
other's thoughts. If each stream ran its own loop, the order of their decisions
would depend on network latency: runs would not be reproducible even with a
fixed seed and recorded responses, and replays would miss requests.

## Decision

The engine runs in rounds with three phases:

1. **Decide** — streams choose their next focus one after another, in a fixed
   order; each claim is immediately visible to the streams after it.
2. **Call** — the model calls of all streams run concurrently.
3. **Integrate** — results are written into the tree in the same fixed order.

Each stream draws from its own generator seeded with `f"{seed}:{index}"`.

## Consequences

- With a deterministic model (scripted or replayed) a run is exactly
  reproducible, parallel streams included.
- A round waits for its slowest call, which costs some wall-clock time compared
  with free-running streams — an acceptable price for reproducibility.
- All shared state is mutated only between awaits, so no locks are needed.
