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
hypothesis: Replacing the twelve-unit GELU MLP with a seven-unit learned GLU will reduce the model from 866 to 847 parameters while retaining at least 99% accuracy, because seven independent output directions span the entire LayerNorm-visible residual quotient and multiplicative gates provide greater nonlinear efficiency per direction.
change: Add a generic gauge-fixed output projection, replace the GELU MLP with seven learned value/gate pairs, update ambient optimization for its columns, and set the default gated width to seven.
mechanism: Rank-complete sigmoid-gated MLP bottleneck
evidence_used: Tying the MLP readout collapsed accuracy to 37.01%, showing that independent output directions are load-bearing; this design preserves seven independent directions—the maximum observable rank after common-mode quotienting—while challenging the assumption that twelve separate GELU features are required.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 847, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero weight coordinate in each of the seven GLU value rows will reduce the model from 847 to 840 learned parameters while retaining at least 99% accuracy, because independently scaling a value row and inversely scaling its output-projection column leaves the learned MLP function unchanged.
change: Gauge-fix the seven GLU value-row scales at their largest initialized coordinates, reconstruct those coordinates from checkpointed buffers, and optimize the remaining weights in ambient space.
mechanism: GLU value-channel scale quotient
evidence_used: The seven-unit GLU achieved 99.94% accuracy at 847 parameters while preserving seven independent output directions; the proposed change retains every direction and removes only seven independent continuous scaling redundancies between value rows and their corresponding output columns.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4512, "parameters": 840, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one nonzero coordinate of a value-projection row will reduce the model from 847 to 846 learned parameters while retaining at least 99% accuracy, because inverse scaling of the corresponding attention output-projection column preserves the learned function exactly.
change: Anchor the largest initialized coordinate of the first value row, omit it from trainable coordinates, reconstruct it during forward passes, and optimize the remaining row coordinates in ambient space.
mechanism: Single attention value/output scale quotient
evidence_used: The analogous anchored Q/K scale gauges retained 99.96% and 99.93% accuracy; applying one value/output gauge isolates a single exact redundancy, whereas simultaneously gauge-fixing all seven GLU channels collapsed accuracy to 45.12%.
result: training did not finish within the verification time limit

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
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced



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
