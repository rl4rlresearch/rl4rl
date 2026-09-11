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
verified_results: {"accuracy": 0.9998, "parameters": 1570, "training_steps": 20999}
prior_hypothesis: Anchoring the final input coefficient of a second `fc1` output row will reduce the verified model from 1,571 to 1,570 learned parameters while retaining at least 99% accuracy after 21,000 training steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one query-projection coefficient at the initialization scale will reduce the model from 1,571 to 1,570 learned parameters while retaining at least 99% accuracy after 21,000 training steps, because inverse rescaling of the corresponding query and key channels leaves attention scores unchanged.
change: Split the first query-projection row from the dense QKV matrix, represent seven of its coefficients as learned parameters, and fix its eighth coefficient at 0.02 while preserving standard initialization for every learned coefficient.
mechanism: Query–key inverse-scale gauge anchoring
evidence_used: The current 1,571-parameter model achieved 99.76% accuracy, whereas extending positional-row and attention-output shift gauges failed at 84.96%, 70%, and 40.36%; this tests a different exact attention symmetry without further extending those sensitive parameterizations.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7288, "parameters": 1570, "training_steps": 20999}

RECENT RESULT
hypothesis: Anchoring the final input coefficient of a second `fc1` output row will reduce the verified model from 1,571 to 1,570 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Store seven learned coefficients for each of the final two `fc1` rows and reconstruct their eighth coefficients as implicit zeros; the existing learned biases retain full expressivity on LayerNorm outputs.
mechanism: Second MLP-input LayerNorm affine-hyperplane gauge
evidence_used: The 99.76%-accurate 1,571-parameter design already anchors one `fc1` row coefficient this way. Extending that established parameterization to an adjacent row is better supported than the positional, attention-output, and query-scale reductions that achieved 84.96%, 40.36%, and 72.88%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1570, "training_steps": 20999}

RECENT RESULT
hypothesis: Anchoring the final input coefficient of a third `fc1` output row will reduce the verified model from 1,570 to 1,569 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Store seven learned coefficients for each of the final three `fc1` rows and reconstruct their eighth coefficients as implicit zeros.
mechanism: Third MLP-input LayerNorm affine-hyperplane gauge
evidence_used: Extending this parameterization from one anchored `fc1` row to two achieved 99.98% accuracy at 1,570 parameters, substantially outperforming the failed positional, attention-output, and query-scale reductions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2983, "parameters": 1569, "training_steps": 20999}

RECENT RESULT
hypothesis: Anchoring the final input coefficient of the first `fc1` output row, while retaining the two successful final-row anchors, will reduce the model from 1,570 to 1,569 parameters and retain at least 99% accuracy.
change: Store seven coefficients for the first and final two `fc1` rows, reconstructing each eighth coefficient as an implicit zero while leaving the intervening nine rows dense.
mechanism: Nonadjacent MLP-input LayerNorm affine-hyperplane gauge
evidence_used: Two anchored final `fc1` rows achieved 99.98% accuracy at 1,570 parameters, whereas anchoring the adjacent third-from-last row fell to 29.83%; testing a nonadjacent row isolates whether that failure was row-specific under the fixed initialization.
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
