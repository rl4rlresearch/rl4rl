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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9994, "parameters": 847, "training_steps": 4999}
prior_hypothesis: Replacing the twelve-unit GELU MLP with a seven-unit learned GLU will reduce the model from 866 to 847 parameters while retaining at least 99% accuracy, because seven independent output directions span the entire LayerNorm-visible residual quotient and multiplicative gates provide greater nonlinear efficiency per direction.

## Recent verification evidence

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Reusing one seven-feature learned basis for both values and gates, with a learned circulant gate transform, will retain at least 99% accuracy while reducing the model from 847 to 805 parameters.
change: Replace the GLU’s fourteen independent input projections with seven shared projections, seven circulant mixing coefficients, and seven independent gate biases while preserving the established downstream initialization stream.
mechanism: Circulant shared-feature GLU
evidence_used: The verified seven-unit GLU reaches 99.94% accuracy at 847 parameters, showing that seven nonlinear output directions suffice; this tests the load-bearing assumption that every direction also requires a separate learned gate projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0508, "parameters": 805, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero value-projection coordinate per GLU unit removes seven exact value/output scaling redundancies, reducing parameters from 847 to 840 while retaining at least 99% accuracy.
change: Store one initialization-preserving value-row coordinate as a buffer for each GLU unit, reconstruct it during forward, and optimize only the remaining ambient coordinates.
mechanism: Per-unit GLU scaling-gauge fixing
evidence_used: The independent seven-unit GLU achieved 99.94% accuracy at 847 parameters, while sharing value and gate features collapsed accuracy to 5.08%; this patch preserves every independent value and gate projection and removes only exact GLU scaling gauges.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5592, "parameters": 840, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the single query-offset scalar at zero will reduce the verified model from 847 to 846 parameters while retaining at least 99% accuracy.
change: Replace the zero-initialized learned query bias with a nonpersistent zero buffer, preserving the forward computation at initialization and the random initialization stream.
mechanism: Bias-free narrow Q/K attention
evidence_used: The 847-parameter model achieved 99.94% accuracy, while reductions that constrained independent GLU projections failed; this instead tests one isolated attention scalar without altering the verified GLU capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Fixing one zero-deviation final LayerNorm scale at its initialization value of 1 will reduce the verified model from 847 to 846 parameters while retaining at least 99% accuracy.
change: Learn three of the four readout-visible LayerNorm scales and materialize the fourth as a fixed unit scale.
mechanism: Single-coordinate final-normalization scale anchor
evidence_used: The 847-parameter model reached 99.94% accuracy, while constraints on independent GLU projections caused large accuracy losses; this isolates a single readout-scale degree of freedom without changing GLU or attention capacity.
result: training did not finish within the verification time limit



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
