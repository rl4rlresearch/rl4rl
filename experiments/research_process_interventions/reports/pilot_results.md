# Eight-run pilot: results and research trajectories

This file reports what the completed pilot actually produced. It is a record of
the generated ideas, architecture paths, execution outcomes, and measured
values. It is not an experiment plan.

## Bottom line

The eight Modal jobs completed successfully as workflows. Across two research
frameworks and four intervention cells, the agents used all 32 proposal
opportunities:

- AutoResearch generated and trained 16 of 16 proposals.
- OpenEvolve generated 16 proposals; 12 were valid and trained, three were
  rejected as within-run architecture duplicates, and one failed Architecture
  IR validation.
- The 28 trained proposals were 28 structurally distinct architecture graphs.
- Counting the repeated seed evaluation in every run, there were 36
  training-and-evaluation executions.
- Every trained architecture passed execution and transformer-runtime validity.
- Every seed and every valid proposal received public exact-match accuracy 0.0
  and public search score 0.0.

The pilot therefore demonstrates that the generation, validation, training,
logging, and artifact pipeline runs end to end. It does not demonstrate that
one intervention improves research quality. The public metric has a complete
floor effect, the sample is one run per cell, and the portfolio-memory exposure
was not meaningfully populated.

## Conditions and run identities

| Cell | Memory shown to agent | Deliberation prompt |
|---|---|---|
| RD0 | Sequential | Neutral review |
| RD1 | Sequential | Assumption challenge at opportunities 1–4 |
| RD2 | Portfolio | Neutral review |
| RD3 | Portfolio | Assumption challenge at opportunities 1–4 |

| Framework | Cell | Completed run ID | Valid proposals |
|---|---:|---|---:|
| AutoResearch | RD0 | ar-modal-rd0-20260821-03 | 4 / 4 |
| AutoResearch | RD1 | ar-modal-rd1-20260821-02 | 4 / 4 |
| AutoResearch | RD2 | ar-modal-rd2-20260821-02 | 4 / 4 |
| AutoResearch | RD3 | ar-modal-rd3-20260821-02 | 4 / 4 |
| OpenEvolve | RD0 | oe-modal-rd0-20260821-02 | 3 / 4 |
| OpenEvolve | RD1 | oe-modal-rd1-20260821-06 | 2 / 4 |
| OpenEvolve | RD2 | oe-modal-rd2-20260821-02 | 3 / 4 |
| OpenEvolve | RD3 | oe-modal-rd3-20260821-02 | 4 / 4 |

All runs used the same conventional two-block causal decoder seed, the same
seed bundle, a Tesla T4, the non-scientific smoke_train_cuda_v2 profile, ten
training steps, 160 training examples, and the smoke_eval_v1 public evaluation
with 24 synthetic addition cases. Parameter count is descriptive and was not
used for selection. Development losses below are also descriptive; the
controller-visible selection values were public accuracy and public search
score.

## Aggregate numerical results

For valid proposals, public accuracy and public search score were 0.0 in every
cell. The table therefore includes the lowest observed development loss only to
show that training was not numerically identical. It must not be interpreted as
an architecture ranking under this smoke profile.

| Framework | Cell | Generated | Trained | Invalid or duplicate | Lowest proposal development loss | Selected endpoint |
|---|---:|---:|---:|---:|---:|---|
| AutoResearch | RD0 | 4 | 4 | 0 | 9.731190 | Proposal 4, by valid plateau acceptance |
| AutoResearch | RD1 | 4 | 4 | 0 | 5.231192 | Proposal 4, by valid plateau acceptance |
| AutoResearch | RD2 | 4 | 4 | 0 | 5.720516 | Proposal 4, by valid plateau acceptance |
| AutoResearch | RD3 | 4 | 4 | 0 | 9.695308 | Proposal 4, by valid plateau acceptance |
| OpenEvolve | RD0 | 4 | 3 | 1 invalid | 5.351149 | Original seed |
| OpenEvolve | RD1 | 4 | 2 | 2 duplicates | 9.623280 | Original seed |
| OpenEvolve | RD2 | 4 | 3 | 1 duplicate | 9.639715 | Original seed |
| OpenEvolve | RD3 | 4 | 4 | 0 | 6.926411 | Original seed |

The repeated seed had 6,080 parameters, best development loss 9.750893, final
training loss 7.039633, public accuracy 0.0, and public search score 0.0.

## AutoResearch trajectories

AutoResearch formed a chain in each run. Because its configured mechanics-only
greedy rule accepts valid plateau moves, every valid zero-score proposal became
the parent of the next proposal. In the tables, “best loss” is best development
loss during the ten training steps and “final loss” is the final training loss.
All listed proposals were execution-valid, transformer-valid, eligible, and
accepted; all had public accuracy 0.0 and search score 0.0.

### RD0 — sequential memory, neutral review

Path: seed → dual retrieval paths → gated computation → more depth →
sinusoidal positioning.

| Step | Candidate | Parent | Generated architecture and hypothesis | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | 44d0007e5834 | ba4ac51ced47 | Dual-Path Alignment and Carry Decoder: separate one-head causal paths should specialize for digit alignment and carry retrieval. | 7,616 | 9.820134 | 7.120163 |
| 2 | cdf962fa5db2 | 44d0007e5834 | Dual-Path Gated Digit Decoder: a SiLU-gated first feed-forward block should combine aligned digit and carry features multiplicatively. | 8,384 | 10.025194 | 7.349373 |
| 3 | 57789e6ba2cd | cdf962fa5db2 | Dual-Path Three-Stage Carry Decoder: a third block should add a stage for integrating retrieved digits into carry information. | 11,008 | 9.731190 | 7.026487 |
| 4 | a25b1d1ec162 | 57789e6ba2cd | Dual-Path Sinusoidal Alignment Decoder: sinusoidal positions should expose stable offsets between operands and reversed outputs. | 10,448 | 10.229189 | 7.244780 |

### RD1 — sequential memory, assumption challenge

Path: challenge insufficient-depth assumption → challenge ordinary
feed-forward computation → challenge learned positions → challenge loss of
original input cues.

| Step | Candidate | Parent | Challenged assumption and resulting architecture | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | 95bf8be0b015 | ba4ac51ced47 | Challenged whether two blocks provide enough algorithmic stages; proposed a four-stage causal decoder. | 11,328 | 9.203272 | 6.498392 |
| 2 | 85a552c6aa32 | 95bf8be0b015 | Challenged whether generic GELU stages are sufficient; replaced all four feed-forward blocks with SiLU-gated blocks. | 14,400 | 9.539908 | 6.630878 |
| 3 | a11f5f73ba89 | 85a552c6aa32 | Challenged whether learned absolute positions expose the required alignment; changed to sinusoidal positions. | 13,840 | 10.050825 | 7.124599 |
| 4 | dff1c7a1edbf | a11f5f73ba89 | Challenged whether ordinary residuals preserve exact operand cues; reinjected the original token-plus-position representation before blocks two through four. | 15,376 | 5.231192 | 5.109486 |

### RD2 — portfolio memory, neutral review

Path: gated computation → parallel retrieval → more depth → hybrid positions.

| Step | Candidate | Parent | Generated architecture and hypothesis | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | bdc618fd5bb2 | ba4ac51ced47 | Gated SiLU AdderBoard Decoder: multiplicative digit-and-carry interactions should outperform GELU blocks. | 7,616 | 10.021337 | 7.235056 |
| 2 | 80b1ad97243a | bdc618fd5bb2 | Parallel Retrieval Adder Decoder: two full-width causal branches should specialize to the two operand streams. | 9,152 | 10.065940 | 7.392326 |
| 3 | 7e57c0815a5b | 80b1ad97243a | Depth-Three Parallel Retrieval Decoder: three stages should separate retrieval, digit computation, and carry integration. | 12,544 | 9.840044 | 7.109300 |
| 4 | 2aa8d64ad5e2 | 7e57c0815a5b | Hybrid-Position Parallel Retrieval Decoder: learned and sinusoidal positions together should provide stable offsets plus lookup flexibility. | 13,056 | 5.720516 | 4.152149 |

### RD3 — portfolio memory, assumption challenge

Path: challenge GELU sufficiency → challenge two-head sufficiency → challenge
two-block sufficiency → challenge learned-position sufficiency.

| Step | Candidate | Parent | Challenged assumption and resulting architecture | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | 3b68e4b3a019 | ba4ac51ced47 | Challenged ordinary GELU computation; proposed gated carry-sensitive feed-forward blocks. | 7,616 | 10.021337 | 7.235056 |
| 2 | cc279fcbd292 | 3b68e4b3a019 | Challenged whether two heads provide enough separate retrieval channels; changed both blocks to four heads. | 7,616 | 10.036663 | 7.257160 |
| 3 | 69a3ebbfa0f4 | cc279fcbd292 | Challenged whether two transformer blocks provide enough stages; added a third four-head gated block. | 11,008 | 9.695308 | 6.827663 |
| 4 | 3f06dd8f2a55 | 69a3ebbfa0f4 | Challenged whether learned absolute positions expose operand-output offsets; changed to sinusoidal positions. | 10,448 | 10.597857 | 7.609538 |

## OpenEvolve trajectories

OpenEvolve formed a star rather than a chain: every proposal used the original
seed as parent. Since no child improved public accuracy or public search score,
the seed remained the best program in all four runs. “Rejected duplicate”
means the generated text and name could differ but the normalized Architecture
IR duplicated a graph already evaluated in that run.

### RD0 — sequential memory, neutral review

| Step | Candidate | Generated architecture and hypothesis | Outcome | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | 7a2d6126 | Learned Depth Mixture: softmax-mix one-block and two-block states to retain local digits and deep carry features. | Valid; accuracy/score 0.0 | 6,082 | 9.899629 | 7.202801 |
| 2 | 93148225 | Input-Context Fusion: concatenate the original token-position state with the final contextual state. | Valid; accuracy/score 0.0 | 6,592 | 5.351149 | 4.607654 |
| 3 | 6de7cec8 | Heterogeneous Attention Mix: route between one-head and four-head causal attention. | Valid; accuracy/score 0.0 | 7,106 | 9.639715 | 6.826387 |
| 4 | 1bec4c2a | Gated Attention Residual: sigmoid-gate carry-context injection. | Invalid IR: sigmoid_gate had two inputs but requires exactly three | — | — | — |

### RD1 — sequential memory, assumption challenge

| Step | Candidate | Challenged assumption and generated architecture | Outcome | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | 918d380a | Challenged one fixed head partition for both alignment and carry; proposed one-head/four-head softmax routing. | Valid; accuracy/score 0.0 | 7,106 | 9.623280 | 6.819330 |
| 2 | 990e1891 | Rechallenged the fixed two-head interface; generated another multi-granularity router. | Rejected duplicate of an already evaluated graph | — | — | — |
| 3 | f9647288 | Challenged deepest-state-only readout; proposed learned routing between shallow and deep states. | Valid; accuracy/score 0.0 | 6,082 | 9.899629 | 7.202801 |
| 4 | 4e667c9f | Rechallenged mandatory serial depth; generated another adaptive shallow/deep mixture. | Rejected duplicate of an already evaluated graph | — | — | — |

### RD2 — portfolio memory, neutral review

| Step | Candidate | Generated architecture and hypothesis | Outcome | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | bcfc03b4 | Heterogeneous Attention Router: mix one-head and four-head attention before carry propagation. | Valid; accuracy/score 0.0 | 7,106 | 9.639715 | 6.826387 |
| 2 | 8aab0c28 | Parallel Head Routing: another one-head/four-head first-block mixture. | Rejected duplicate of an already evaluated graph | — | — | — |
| 3 | 5c90ba70 | Adaptive Depth Router: mix embedding-level and successive decoder states. | Valid; accuracy/score 0.0 | 6,083 | 10.112799 | 7.393988 |
| 4 | 5da88f0d | Dual Granularity Attention Mixer: fixed mixtures of one-head and four-head attention in both blocks. | Valid; accuracy/score 0.0 | 8,128 | 9.771584 | 6.875888 |

### RD3 — portfolio memory, assumption challenge

| Step | Candidate | Challenged assumption and generated architecture | Outcome | Parameters | Best loss | Final loss |
|---:|---|---|---|---:|---:|---:|
| 1 | c0d7341e | Challenged one attention pathway serving alignment and carry; proposed routed one-head/four-head branches in both blocks. | Valid; accuracy/score 0.0 | 8,132 | 9.774013 | 6.880223 |
| 2 | 5d2d47cf | Challenged deepest-state-only readout; proposed learned shallow/deep cross-depth fusion. | Valid; accuracy/score 0.0 | 6,592 | 7.187851 | 6.219342 |
| 3 | 7b668774 | Challenged preservation of token-local information in the deepest state; fused positional, intermediate, and deep states. | Valid; accuracy/score 0.0 | 6,848 | 6.926411 | 5.058044 |
| 4 | 6a80dfd7 | Challenged homogeneous attention for carry and local dependencies; proposed one-head/four-head routing. | Valid; accuracy/score 0.0 | 7,106 | 9.639715 | 6.826387 |

## What visibly changed across conditions

The generated paths were not identical. AutoResearch RD0 explored parallel
retrieval before gating and depth; RD1 began with depth and ended with repeated
input fusion; RD2 began with gating and ended with hybrid positional encoding;
RD3 moved through gating, more heads, more depth, and sinusoidal positions.
OpenEvolve concentrated heavily on routing: neutral RD0 also tried input
fusion, challenged RD1 alternated head routing and depth routing, portfolio RD2
mostly varied routing granularity, and combined RD3 included the broadest set
of fusion ideas.

Those are descriptive trajectory differences, not treatment effects. With one
stochastic run per cell, the intervention is confounded with ordinary sampling
variation.

## Instrumentation findings that limit interpretation

Two issues are as important as the generated architectures:

1. Portfolio memory was effectively empty or degenerate. AutoResearch exposure
   records contained four not_available placeholders at every opportunity.
   OpenEvolve portfolio records exposed the seed in both the current/recent and
   distant-alternative slots, while valid-failure and abandoned-direction slots
   remained empty. The pilot therefore did not provide a genuinely diverse
   portfolio whose causal effect could be measured.
2. AutoResearch wrote four treatment-exposure records per run but its
   research_process/decisions.jsonl files were empty in all four conditions.
   Its proposal hypotheses and challenged-assumption fields survive in lineage
   and candidate IR metadata, but the common decision-event instrumentation is
   missing. OpenEvolve did write four decision records per run.

OpenEvolve also wrote each exposure twice per opportunity. That duplication did
not create extra proposals, but it should be corrected before process-count
metrics are computed from exposure rows.

## Correct interpretation

What succeeded:

- all eight orchestrated jobs reached successful terminal completion;
- all 32 LLM proposal opportunities produced trajectory artifacts;
- 28 proposed graphs plus eight repeated seeds trained on CUDA;
- runtime and transformer-validity checks passed for every trained candidate;
- assumptions, hypotheses, parent links, failures, and public outcomes are
  recoverable from the artifacts.

What did not succeed scientifically:

- the ten-step smoke budget produced zero public accuracy everywhere;
- public scores could not distinguish ideas or drive evidence-based revision;
- portfolio memory did not contain a real portfolio;
- AutoResearch decision-event logging was absent;
- one run per framework-condition cell cannot estimate intervention effects.

The honest result is an end-to-end infrastructure pilot with rich qualitative
idea traces, not a completed experiment about how memory or assumption
challenges change research.

## Artifact provenance

The source artifacts are stored in the Modal volume
rl4rl-architecture-artifacts under runs/<run-id> for each run ID listed above.
The locally downloaded, SHA-256-verified subset used to prepare this report is
under outputs/process/final-pilot-trajectories-20260821. That output directory
is intentionally ignored by Git because it contains approximately 2.3 MB of
machine-generated run artifacts; this Markdown file is the shareable,
human-readable extraction.
