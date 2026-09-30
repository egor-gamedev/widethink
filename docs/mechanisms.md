# From the brain to the harness

The harness transplants the thought stream of the digital brain
(`digital_brain/neurons/thought.py`, with the branching mock-up
`runtime/thinking.py`). In the brain a node of thought is a concept neuron; here
it is an idea or a question formulated by the model. Link strength becomes the
model's estimate of how directly one idea follows from another, and closeness of
ideas is measured with embeddings.

## One tick, one thought

In the brain, each tick the stream decides only one thing — what to think about.
The network does the rest with the same pass it uses for perception. In the
harness, each step the stream decides what to think about, and one model call
thinks it.

The brain measures time in ticks of 0.1 s; a thought dwells for 3–8 ticks, 5.5 on
average. The harness measures time in thoughts, so constants defined per tick are
converted with `TICKS_PER_THOUGHT = 5.5` (`config.py`).

## Mapping

| Mechanism | In the brain | In the harness | Default | Code |
|---|---|---|---|---|
| Focus and reinstatement | Lights the focus and the 3 neurons it is made of (1 when weak) | One call about one idea; the path to it and the 3 most relevant context items go with it (1 when strength < 0.5) | `items=3`, `weak_items=1`, `weak_below=0.5` | `engine._reinstate`, `fade.reinstated_items` |
| Association step | One of the 6 strongest links, drawn with weight score² | The model proposes up to 6 next ideas with link strengths; the harness draws by lottery | `max_candidates=6`, `square_weights=True` | `selection.draw` |
| Score of a candidate | link × (1 − fatigue) × pull(value), ×1.2 for memory traces | link × (1 − redundancy) × pull(value), ×1.2 if grounded in the context | `grounded_bonus=1.2` | `selection.score` |
| Thought within a thought, return | Stack of enclosing thoughts; a finished branch returns to the parent for another link | Explicit stack per stream; return to the parent for another child | — | `stream._advance` |
| Satiation ("bored") | +0.12 per repeated tick, −0.06 otherwise, max 0.8; bored at 0.45 | Repetition of a thought = how much its proposals repeat ideas in the tree; +0.66 per fully repetitive thought, −0.33 per novel one | `gain=0.66`, `recovery=0.33`, `satiated=0.45`, `max_fatigue=0.8` | `fatigue.update_fatigue` |
| Inhibition of return | 60 ticks without the recent focus | Already-considered list shown to the model; duplicates of considered ideas pruned | `window=None` (session) | `inhibition.py`, `engine._integrate`, `stream._inhibited` |
| Fading | ×0.86 per level; exhausted below 0.22; at most 7 levels; mock-up: × min(1, 0.4 + link) | Same; strength also sets how many proposals a thought makes | `factor=0.86`, `min_strength=0.22`, `max_depth=7`, `link_modulation=0.4` | `fade.py` |
| Pull to value (M3) | weight × exp(value), value clipped to [−0.5, 1]; about 20:1 good vs bad | The critic (inline or separate) scores ideas by fit to this user | `pull=1.0`, `floor=−0.5`, `ceiling=1.0` | `value.value_pull` |
| Capture by surprise | Surprise 0.2 above its moving average hijacks the stream, not more often than every 15 ticks | A thought reporting a contradiction or unexpected fact makes it the next thought of that stream | `threshold=0.2`, `baseline_rate=0.246`, `refractory=3` | `capture.SurpriseMonitor` |
| Emergence, cued by perception | Half of new thoughts are prompted by what is perceived | Half of new branches start from a detail of the user's context: "What does X imply for this task?" | `cued=0.5` | `emergence.choose_emergence`, `stream._cue` |
| Emergence from memory | A trace surfaces with weight (0.05 + salience + recency) × pull | Unexplored ideas of this run and findings of past runs surface the same way (and less readily if they repeat) | `recency_scale=5.45` | `emergence.recall_weight`, `stream._emerge` |
| Memory grows by thinking; sleep | A thought-about trace counts as rehearsed; sleep replays and links old and new | Valuable, evidenced findings are stored; `sleep()` merges duplicates, links related findings, fades unused ones | `FADE_PER_SLEEP=0.97` | `memory/store.py` |
| Parallel branches | Only in the mock-up: `branching = 2` | Several streams over one shared tree; they see each other's thoughts | `parallel=2` | `engine._round` |

## Derivations

- **Fatigue per thought.** The brain adds 0.12 each tick the same pattern repeats.
  A thought lasts 5.5 ticks, so a fully repetitive thought adds 0.66 and bores the
  branch at once (0.66 ≥ 0.45); a half-repetitive one adds 0.66·0.5 − 0.33·0.5 =
  0.165 and bores on the third. Recovery 0.06 × 5.5 = 0.33.
- **Capture baseline.** The brain's moving average moves 5% per tick; per thought
  that is 1 − 0.95^5.5 ≈ 0.246. The 15-tick gap between captures is 15 / 5.5 ≈ 3
  thoughts.
- **Recency.** The brain's recency term 1 / (1 + age / 30 ticks) becomes
  1 / (1 + age / 5.45 thoughts).
- **Value ratio.** With squared weights, the pull ratio between value 1 and value
  −1 (clipped to −0.5) is (e¹ / e^−0.5)² = e³ ≈ 20.
- **Inhibition window.** 60 ticks ≈ 11 thoughts. The brain's stream never ends, so
  an idea may return later; a run is bounded, and re-thinking an idea within it
  only wastes budget, so the default window is the whole run.

## What differs from existing methods

- **Breadth over the user's context, not only over solutions.** Tree of Thoughts,
  self-consistency and parallel reasoning branch over ways to answer the question
  as asked. Here half of the new branches start from details of the user's
  context and ask what is non-standard about them.
- **Stopping rules from biology.** Branches close not only by quality but by
  repetition (satiation), strength (fading), recency (inhibition of return) and
  interruption (capture) — aimed directly at the "tunnel vision" of a single
  chain of thought.
- **Streams that see each other.** Parallel streams share the tree and inhibit
  each other's duplicates instead of rediscovering the same idea independently.
- **Diversity from the harness, not from sampling.** The lottery, not the model's
  temperature, makes thinking vary, so the method also works with models that do
  not accept sampling parameters.

All defaults are the brain's, not tuned values. Phase 1 of the roadmap calibrates
them on development tasks; the ablations of Phase 3 measure what each contributes.
