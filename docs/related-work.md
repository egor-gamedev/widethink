# Related work

Today labs buy breadth of thinking with compute at answer time (test-time
compute): a longer chain of thought, or more chains in parallel. All of these
forms search for the best answer to the question *as it was asked*.

| Form | Where it is used | What is missing for hidden requirements |
|---|---|---|
| Chain of thought / reasoning modes | Reasoning modes of current models, trained with RL | The first steps lock thinking into one path — "tunnel vision" ([ParaThinker](https://arxiv.org/abs/2509.04475)) |
| Sampling and voting (self-consistency, Wang et al., 2022) | Standard practice in research and products | Samples resemble each other; voting selects the most typical answer |
| Tree search with evaluation (Tree of Thoughts, Yao et al., 2023; MCTS variants) | Research | Branches are pruned by quality, not by repetition; costly in tokens |
| Parallel reasoning with synthesis ([Adaptive Parallel Reasoning](https://bair.berkeley.edu/blog/2026/05/08/adaptive-parallel-reasoning/), deep-thinking modes) | Research 2025–2026, flagship "deep" modes | Paths do not see each other and can duplicate; models drift back to sequential thinking |
| Multi-agent research systems (a lead agent with helpers) | Research agents | Subtasks are cut from the question as worded |
| Diversity prompting ([Verbalized Sampling](https://arxiv.org/abs/2510.01171)) | Creative generation; ×1.6–2.1 diversity | Diverse answers are not the same as answers that fit the user |

The root of the problem is known: models are biased toward the most typical
answer, partly because human raters prefer the familiar during training (the
typicality-bias finding of the Verbalized Sampling authors).

## Where widethink differs

1. **Breadth over the user's context.** Existing methods branch over solutions to
   the question as formulated; widethink also branches over what is special about
   this user, starting half of its new branches from details of the context.
2. **Biological stopping rules.** Satiation, inhibition of return, fading and
   capture close branches by repetition and interruption, not only by quality.
3. **Streams that see each other.** Parallel streams share one tree and inhibit
   each other's duplicates.
4. **Memory of findings.** Non-standard solutions that worked are kept and can
   surface in related tasks; offline consolidation links them.
5. **A benchmark for the failure mode itself**, with controls that punish
   inventing requirements, evaluated at equal token budgets.

## What will not work, and what will

- Labs build this kind of thinking into models through training, not external
  harnesses; a harness is a source of ideas and training data for them, not a
  product.
- Large companies generally do not consider unsolicited ideas; published,
  reproducible results are what they read.
- The first question will be: "with the same number of tokens, does a plain
  model do as well?". Without equal-budget comparisons a result will not be
  taken seriously — hence the design of the benchmark.

## References

- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language
  Models*, 2022.
- Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language
  Models*, 2023.
- *ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time
  Compute*, [arXiv:2509.04475](https://arxiv.org/abs/2509.04475).
- *Adaptive Parallel Reasoning*, BAIR blog, May 2026,
  [link](https://bair.berkeley.edu/blog/2026/05/08/adaptive-parallel-reasoning/).
- *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity*,
  [arXiv:2510.01171](https://arxiv.org/abs/2510.01171).
- Gebru et al., *Datasheets for Datasets*, 2018 — the template for the benchmark's
  datasheet.
