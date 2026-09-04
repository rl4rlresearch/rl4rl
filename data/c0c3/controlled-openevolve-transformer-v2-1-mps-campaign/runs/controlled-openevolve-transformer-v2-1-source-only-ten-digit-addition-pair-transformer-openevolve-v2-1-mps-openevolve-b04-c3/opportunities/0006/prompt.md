# Optimize a transformer for 10-digit addition

You are an autonomous ML engineer improving the source code for an
autoregressive transformer that adds two 10-digit numbers.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% accuracy under the fixed verification process. A
smaller implementation is useful only when it meets that accuracy requirement.
Every submitted implementation is trained from a fresh initialization.

## Learned-model requirement

Produce a smaller trained autoregressive transformer, not a hand-coded addition
program. The submitted implementation must:

- have nonzero trainable parameters;
- contain and use at least one learned causal self-attention module;
- map token inputs to token logits through the learned model;
- train from a fresh initialization during verification;
- write both `checkpoints/best.pt` and a positive-step `checkpoints/last.pt`;
- keep source code unchanged while training; and
- use the protected generic decoding interface exactly as supplied.

Do not implement or embed decimal arithmetic, carry propagation, place-value
rules, digit lookup tables, finite-state addition transitions, fixed answer
rules, or input-dependent Python logic that directly computes the sum. Do not
hide such a solver in model generation, token processing, training, or saved
weights. Do not add dummy or zero-length parameters to disguise a fixed
algorithm as a learned model.

Do not modify protected files. Do not perform post-training state-dictionary
surgery, substitute a different saved model, truncate weights after training,
or report a parameter count that differs from the submitted model.

## Work boundaries

Minimize parameters. Required result: accuracy >= 0.99.
Editable source files: src/model.py, src/train.py.
Results reported after each verification: accuracy, parameters, training_steps.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or verification yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining the tied token and positional embeddings to a learned six-dimensional subspace will retain at least 99% accuracy while reducing parameters from 1,644 to 1,418.
change: Replace full-width token and positional tables with rank-6 learned codes projected into the existing eight-dimensional transformer, and reuse the resulting token embeddings for logits.
mechanism: Rank-6 shared embedding factorization
evidence_used: The current eight-dimensional model reaches 99.96% accuracy, leaving substantial margin for a two-dimensional embedding bottleneck without reducing attention or MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.07690000000000001, "parameters": 1418, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will retain at least 99% accuracy while reducing learned parameters from 1,644 to 1,627.
change: Preserve the full-width token and positional embeddings and reduce the transformer block’s feed-forward hidden width by one unit.
mechanism: Single-unit MLP width reduction
evidence_used: The full-width `d_model=8, d_ff=12` model achieved 99.96% accuracy, while rank-6 embedding factorization collapsed to 7.69%; this motivates preserving embedding expressivity and testing a small reduction in a different subsystem.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7493000000000001, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the mathematically redundant key-projection bias will preserve at least 99% accuracy while reducing parameters from 1,644 to 1,636.
change: Make the packed QKV projection bias-free, then restore learned query and value biases as separate parameters while omitting the key bias.
mechanism: Softmax-invariant key-bias elimination
evidence_used: Reducing MLP width to 11 collapsed accuracy to 74.93%, motivating preservation of effective capacity. A constant key bias adds the same value to every unmasked attention logit in each query row and therefore cancels under softmax.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7166, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the redundant key bias while preserving the baseline constructor RNG stream will achieve at least 99% accuracy with 1,636 parameters.
change: Retain the original packed-linear initialization draw, discard its key-containing bias, learn separate zero-initialized query and value biases, and apply them explicitly.
mechanism: RNG-preserving softmax-invariant key-bias removal
evidence_used: The 1,644-parameter baseline achieved 99.96% accuracy. The previous key-bias removal unexpectedly reached only 71.66%; constructing the packed projection directly without bias changes subsequent seeded initialization, so preserving the baseline RNG stream isolates the mathematically invariant parameter removal.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8083, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing a zero key-bias slice inside the fused linear operation will preserve baseline numerical behavior and achieve at least 99% accuracy with 1,636 parameters.
change: Store only the learned query and value bias slices while passing a reconstructed full-width bias directly to `F.linear`.
mechanism: Fused reconstruction of a softmax-invariant key bias
evidence_used: The 1,644-parameter baseline reached 99.96%. Earlier key-bias removal preserved initialization but reached 80.83% when biases were applied explicitly; retaining the original fused bias computation tests whether that numerical change caused the regression.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4971, "parameters": 1636, "training_steps": 4999}



Use the available technical evidence to choose the most informative next
change. Treat unsuccessful or malformed work as evidence when a useful
subject-level reason is provided. Do not invent missing evidence.

## Response

Return these short metadata lines followed by one or more exact
`SEARCH`/`REPLACE` blocks that together produce one implementation:

`MECHANISM: <a concise free-form name for the computational idea>`

`HYPOTHESIS: <a falsifiable claim grounded in the evidence above>`

`INTENDED_EDIT: <what this patch changes>`

`EVIDENCE: <the most relevant prior result and why it motivates this patch>`

Start each block with `<<<<<<< SEARCH`, put the exact existing lines next, use a
line containing `=======` as the divider, put the replacement lines after it,
and finish the block with `>>>>>>> REPLACE`.

Every `SEARCH` section must be nonempty and match exactly once after earlier
blocks have been applied. All blocks must apply. They may edit either or both
editable files, but together they must describe one implementation ready for
verification. The mechanism name is descriptive, not chosen from a fixed list.
Do not paste whole files, lengthy logs, or routine progress reports outside the
patch.
