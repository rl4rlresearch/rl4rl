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

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9943000000000001, "parameters": 1627, "training_steps": 4999}
prior_hypothesis: Omitting one head-1 query-bias coordinate while preserving its critical final value-bias scalar will produce a 1,627-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1625, "training_steps": 4999}
prior_hypothesis: Omitting a second head-0 query-bias coordinate while retaining all three head-1 query-bias coordinates will produce a 1,625-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1622, "training_steps": 4999}
prior_hypothesis: Tying the retained final head-1 value bias to the corresponding final attention-projection bias will produce a 1,622-parameter model with at least 99% accuracy while preserving both learned bias pathways.

## Recent verification evidence

RECENT RESULT
hypothesis: Tying the two retained head-0 query-bias coordinates will produce a 1,624-parameter model with at least 99% accuracy by preserving learned bias influence in both coordinates.
change: Adopt the verified 1,625-parameter positional, key-bias, value-bias, and 2:3 query-bias layout, then replace head 0’s two independent query biases with one shared scalar.
mechanism: Head-0 rank-one query-bias sharing
evidence_used: The 2:3 query-bias design achieved 99.99% at 1,625 parameters, while zeroing another head-0 coordinate narrowly missed at 98.68%; sharing retains both successful coordinate pathways with one degree of freedom.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9964, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the first two retained head-1 query-bias coordinates will reduce the qualified model to 1,623 parameters while maintaining at least 99% accuracy, because both coordinate pathways remain active and the analogous head-0 sharing retained 99.64% accuracy.
change: Replace two independent head-1 query-bias scalars with one shared scalar while preserving the existing shared head-0 biases, third head-1 query bias, final head-1 value bias, key-bias omissions, and positional quotient.
mechanism: Head-1 rank-one query-bias sharing
evidence_used: The current 1,624-parameter design succeeded by sharing two head-0 query biases, whereas zeroing an additional head-0 bias narrowly failed and reducing head 1 by zeroing a coordinate collapsed accuracy; sharing is therefore the most informative way to reduce head-1 capacity without removing another query pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9298000000000001, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the second and third retained head-1 query-bias coordinates, while keeping its first coordinate independent, will produce a 1,623-parameter model with at least 99% accuracy.
change: Apply the qualified head-0 query-bias sharing and additionally tie the two head-1 query coordinates that do not align with the position-embedding gauge-fixed hidden coordinate.
mechanism: Gauge-aware head-1 query-bias sharing
evidence_used: Head-0 sharing achieved 99.64% at 1,624 parameters, while tying the first two head-1 coordinates fell to 92.98%; testing the alternative head-1 pair preserves the first coordinate independently and isolates whether that coordinate-specific coupling caused the failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6101, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying an exact fixed-norm quotient to one zero-bias head-0 query row in the qualified shared-bias design will produce a 1,623-parameter model with at least 99% accuracy.
change: Adopt the verified 1,624-parameter head-0 query-bias sharing layout, then represent one omitted-bias query row by seven stereographic direction parameters and inversely rescale its paired key row.
mechanism: Per-coordinate query–key scaling gauge quotient
evidence_used: Head-0 bias sharing achieved 99.64% at 1,624 parameters, while both tested head-1 sharing reductions failed; the query–key rescaling symmetry removes a parameter without tying or deleting another sensitive query pathway and preserves initialized attention scores exactly.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2345, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate in the qualified 1,624-parameter design will yield 1,623 parameters and at least 99% accuracy because the downstream `fc1` column retains the removed scaling degree of freedom.
change: Reproduce the verified positional and QKV compactions, then reconstruct the final `ln2` scale as one while learning its other seven scales and all biases.
mechanism: Final-coordinate LayerNorm–MLP scaling gauge quotient
evidence_used: The 1,624-parameter shared head-0 query-bias design achieved 99.64%; prior 1,623 attempts modified fragile attention coordinates, whereas this reduction preserves that qualified attention layout and removes an exact scale redundancy immediately before a free MLP input column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.298, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` bias coordinate at zero will produce a 1,623-parameter model with at least 99% accuracy because `fc1.bias` can absorb its entire downstream effect while the qualified attention layout and all LayerNorm scales remain unchanged.
change: Replace each block’s `ln2` with a compact LayerNorm retaining all eight scales and seven bias coordinates, reconstructing the final bias coordinate as zero.
mechanism: LayerNorm-to-MLP affine-offset quotient
evidence_used: The current 1,624-parameter design achieved 99.64%. Tested 1,623-parameter attention changes and the `ln2` scale quotient failed, motivating a distinct exact affine-offset redundancy that does not alter attention or constrain `ln2` scaling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9967, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,623-parameter design by fixing a second `ln2` bias coordinate at zero will produce a 1,622-parameter model with at least 99% accuracy, because both removed offsets can be absorbed by `fc1.bias`.
change: Reproduce the verified shared head-0 query-bias layout, then retain six learned `ln2` bias coordinates and reconstruct the final two as zero.
mechanism: Incremental LayerNorm-to-MLP bias quotient
evidence_used: The 1,623-parameter design fixing one `ln2` bias coordinate achieved 99.67%; this tests the smallest incremental extension of that successful exact affine-offset redundancy without further modifying fragile attention capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9773000000000001, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second positional-embedding coordinate through the exact tied-embedding translation symmetry will reduce the qualified 1,623-parameter model to 1,622 parameters while maintaining at least 99% accuracy.
change: Adopt the qualified shared head-0 QKV layout and one-coordinate `ln2` bias quotient, then remove a second positional scalar while compensating its initialization in the corresponding token-embedding column.
mechanism: Second token–position translation gauge quotient
evidence_used: The 1,623-parameter design achieved 99.67% with the established one-coordinate positional quotient; unlike the failed second `ln2` bias removal, this change extends a symmetry that preserves initialized hidden inputs and softmax probabilities exactly.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5063, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing the final `fc2` output-bias coordinate will produce a 1,622-parameter model with at least 99% accuracy because any common shift of the residual stream is removed by the final LayerNorm.
change: Reproduce the qualified 1,623-parameter design, then represent `fc2.bias` relative to its final coordinate and reconstruct that coordinate as zero.
mechanism: Residual-stream common-mode bias quotient
evidence_used: The qualified 1,623-parameter design reached 99.67%; unlike the failed second `ln2`-bias and positional removals, this targets an untouched functionally null common-mode direction immediately upstream of the final LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2613, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the retained final head-1 value bias to the corresponding final attention-projection bias will produce a 1,622-parameter model with at least 99% accuracy while preserving both learned bias pathways.
change: Split the compact QKV bias into query and value parameters, then reuse the value scalar as the final projection-bias coordinate instead of learning a separate scalar.
mechanism: Shared value–projection offset quotient
evidence_used: The qualified 1,623-parameter model achieved 99.67% while retaining the critical final head-1 value bias; prior 1,622 designs fixed sensitive coordinates to zero, whereas this sharing removes a redundant constant offset without deleting either pathway.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Building on the verified 1,622-parameter design, tying the penultimate two `ln2` bias coordinates while retaining both pathways will yield 1,621 parameters and at least 99% accuracy.
change: Reproduce the qualified shared head-0 query bias, shared value/projection offset, and final-zero `ln2` bias, then share the two adjacent retained `ln2` bias coordinates instead of fixing the second one to zero.
mechanism: Shared adjacent LayerNorm-offset quotient
evidence_used: The 1,622-parameter value/projection-sharing design achieved 99.92%; independently, sharing two head-0 query biases succeeded at 99.64% where deleting the second pathway missed at 98.68%, motivating sharing rather than the failed second `ln2`-bias deletion.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6233, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the tied embedding’s globally uniform scalar direction from the verified 1,622-parameter design will yield 1,621 parameters while maintaining at least 99% accuracy, because LayerNorm cancels its input effect and softmax cancels its uniform logit shift.
change: Reproduce the qualified shared-query, compact-ln2, and shared value/projection-bias design, then gauge-fix one globally redundant tied-embedding scalar.
mechanism: Tied-embedding global translation gauge quotient
evidence_used: The 1,622-parameter shared value/projection design achieved 99.92%; the prior 1,621 attempt instead constrained sensitive ln2 offsets and fell to 62.33%, motivating an untouched exact invariance outside attention and MLP capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2576, "parameters": 1621, "training_steps": 4999}



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
