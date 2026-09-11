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
verified_results: {"accuracy": 1.0, "parameters": 1555, "training_steps": 4999}
prior_hypothesis: Replacing the two independent key/value heads with one shared four-dimensional learned key/value head will retain at least 99% accuracy while reducing the current model from 1,623 to 1,555 parameters, because separate query heads still provide two operand-routing patterns while addition tokens can share a common content representation.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9991, "parameters": 1542, "training_steps": 4999}
prior_hypothesis: Making `ln1` bias-free will reduce the model from 1,550 to 1,542 parameters while retaining at least 99% accuracy, because its query offset is representable by the retained learned query bias, its key offset cancels under softmax, and its position-independent value offset is representable by the attention output bias.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1622, "training_steps": 4999}
prior_hypothesis: Extending the verified three-column terminal gauge to four `fc2` weight columns will reduce the model to 1,622 parameters while retaining at least 99% accuracy, because the additional all-ones output component is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the qualified terminal-bias gauge to one `fc2` weight column will yield 1,625 parameters and retain at least 99% accuracy, because its omitted all-ones output component is erased by the final LayerNorm while full eight-coordinate AdamW dynamics are preserved.
change: Gauge-fix the terminal MLP bias and first weight column, reconstruct both eight-dimensional tensors during forward passes, and optimize their stored differences using ambient AdamW moments and clipping.
mechanism: Terminal MLP output-direction gauge fixing
evidence_used: The positional-plus-terminal-bias ambient gauge achieved 99.95% accuracy at 1,626 parameters. The new reduction applies the same proven gauge and optimizer treatment to one input-dependent scalar shift from the same terminal MLP, avoiding the earlier unsuccessful attention-output gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1625, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified terminal gauge from one to two `fc2` weight columns will yield 1,624 parameters while retaining at least 99% accuracy, because each omitted all-ones output component is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.
change: Reproduce the successful positional and terminal-bias gauges, gauge-fix the first two terminal MLP weight columns, and optimize all four gauge vectors using ambient-coordinate moments and clipping.
mechanism: Two-column terminal MLP output-direction gauge fixing
evidence_used: The one-column terminal gauge achieved 99.99% accuracy with 1,625 parameters; applying the same exact invariance and proven optimizer treatment to one adjacent column is the smallest supported next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1624, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified terminal gauge from two to three `fc2` weight columns will reduce the model to 1,623 parameters while retaining at least 99% accuracy, because the additional omitted all-ones output component is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.
change: Gauge-fix the first three terminal MLP weight columns and update initialization/storage offsets accordingly; the existing ambient optimizer automatically includes the added gauge vector.
mechanism: Three-column terminal MLP output-direction gauge fixing
evidence_used: The one-column gauge achieved 99.99% accuracy at 1,625 parameters and the two-column extension achieved 99.96% at 1,624, directly motivating the smallest adjacent one-parameter extension.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1623, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-column terminal gauge to four `fc2` weight columns will reduce the model to 1,622 parameters while retaining at least 99% accuracy, because the additional all-ones output component is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.
change: Gauge-fix the first four terminal MLP weight columns, reconstruct them during forward passes, and include every column in the existing ambient-coordinate optimizer.
mechanism: Four-column terminal MLP output-direction gauge fixing
evidence_used: The three-column design achieved 99.95% accuracy with 1,623 parameters, following successful one- and two-column designs; extending the same exact invariance by one adjacent column is the smallest evidence-supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1622, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified four-column terminal gauge to five `fc2` weight columns will reduce the model to 1,621 parameters while retaining at least 99% accuracy, because the additional all-ones output component is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.
change: Gauge-fix the first five terminal MLP weight columns, reconstruct them during forward passes, and include all five columns in the ambient-coordinate optimizer.
mechanism: Five-column terminal MLP output-direction gauge fixing
evidence_used: The four-column terminal gauge achieved 99.96% accuracy with 1,622 parameters after the one-, two-, and three-column variants also exceeded 99%; extending the same exact invariance by one adjacent column is the smallest evidence-supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5892000000000001, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing terminal MLP columns 0–3 and 11 will produce a 1,621-parameter model with at least 99% accuracy, showing whether the failed five-column result was specific to column 4 rather than the number of gauge vectors.
change: Preserve the first four qualified gauges, leave the previously destabilizing fifth column fully learned, and gauge-fix the final `fc2` column with the existing ambient-coordinate AdamW treatment.
mechanism: Nonadjacent fifth terminal output-direction gauge
evidence_used: Four adjacent terminal-column gauges achieved 99.96% at 1,622 parameters, while adding column 4 collapsed accuracy to 58.92%; substituting a nonadjacent column is the smallest controlled test of column-specific optimization sensitivity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8085, "parameters": 1621, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the two independent key/value heads with one shared four-dimensional learned key/value head will retain at least 99% accuracy while reducing the current model from 1,623 to 1,555 parameters, because separate query heads still provide two operand-routing patterns while addition tokens can share a common content representation.
change: Challenge the assumption that each query head requires its own key and value projections. Preserve the full eight-dimensional residual stream, tied lexical interface, two learned query heads, MLP, and qualified gauges, but broadcast one learned key/value head across both attention maps.
mechanism: Multi-query causal attention with a shared learned key/value head
evidence_used: The full-width tied embedding is load-bearing—the rank-seven lexical factorization fell to 3.76%—and the gated-MLP alternative reached only 70.49%, so this patch preserves both. The 1,622-parameter four-gauge design reached 99.96%, while further gauge extensions became unstable, motivating a different reduction in attention redundancy that retains both independently learned routing queries.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1555, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified shared key/value attention with the verified four-column terminal gauge will achieve at least 99% accuracy with 1,554 parameters.
change: Broadcast one learned four-dimensional key/value head across two independently learned query heads, and gauge-fix the terminal MLP bias plus its first four weight columns using full eight-coordinate AdamW dynamics.
mechanism: Shared-key/value multi-query attention with four terminal output gauges
evidence_used: Shared key/value attention with three terminal-column gauges achieved 100% accuracy at 1,555 parameters, while the four-column terminal gauge independently achieved 99.96%; this tests their smallest adjacent combination.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1554, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the four-dimensional shared value bias will reduce the verified 1,554-parameter model to 1,550 parameters while retaining at least 99% accuracy, because softmax-normalized attention propagates this position-independent offset unchanged and the learned output-projection bias already represents the resulting constant residual shift.
change: Delete the redundant shared value-bias parameter and compute values solely with the learned shared value projection, preserving both learned query heads, shared key/value attention, and the verified four-column terminal gauge.
mechanism: Absorb shared value bias into attention output bias
evidence_used: Shared key/value attention achieved 100% at 1,555 parameters and remained at 99.98% when combined with the four-column terminal gauge at 1,554; this isolates a functional redundancy inside that qualified attention design without removing the query bias previously shown to be load-bearing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1550, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified positional gauge to the second position row will reduce the verified 1,550-parameter design to 1,549 parameters while retaining at least 99% accuracy, because an independent all-ones shift at that position is erased by each pre-norm LayerNorm and the final LayerNorm, while full eight-coordinate AdamW dynamics are preserved.
change: Reproduce the verified four-column terminal gauge and bias-free shared value projection, then gauge-fix a second positional row and include it in ambient-coordinate optimization.
mechanism: Second-row positional shift gauge with ambient AdamW
evidence_used: The shared-key/value, four-terminal-gauge, bias-free-value design achieved 99.85% accuracy at 1,550 parameters; the existing first-row positional gauge establishes the same exact invariance and optimizer treatment for an adjacent positional row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5317000000000001, "parameters": 1549, "training_steps": 4999}

RECENT RESULT
hypothesis: The verified shared-key/value, bias-free-value design will retain at least 99% accuracy after removing all eight `ln2` scale parameters, yielding 1,542 parameters, because `ln2` scales initialize to one and their per-coordinate scaling can be learned equivalently by the immediately following `fc1` columns.
change: Replace full key/value heads with one shared four-dimensional learned key/value stream, omit the redundant value bias, and make the second pre-norm non-affine.
mechanism: Parameter-free second pre-norm with shared key/value attention
evidence_used: The shared-key/value design without value bias achieved 99.85% accuracy at 1,550 parameters; this preserves that qualified computation while removing only the scale directly preceding a learned affine layer.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7368000000000001, "parameters": 1542, "training_steps": 4999}

RECENT RESULT
hypothesis: Making `ln1` bias-free will reduce the model from 1,550 to 1,542 parameters while retaining at least 99% accuracy, because its query offset is representable by the retained learned query bias, its key offset cancels under softmax, and its position-independent value offset is representable by the attention output bias.
change: Disable only the eight-dimensional bias of the first pre-attention LayerNorm while preserving its learned scale and all existing attention, MLP, and gauge parameters.
mechanism: Redundant first pre-norm offset removal
evidence_used: The current bias-free-value shared-key/value model achieves 99.85% at 1,550 parameters. Unlike the failed removal of `ln2` scales, this removes a zero-initialized affine offset whose three downstream effects already have exact absorption or invariance mechanisms, making it a controlled test at 1,542 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1542, "training_steps": 4999}



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
