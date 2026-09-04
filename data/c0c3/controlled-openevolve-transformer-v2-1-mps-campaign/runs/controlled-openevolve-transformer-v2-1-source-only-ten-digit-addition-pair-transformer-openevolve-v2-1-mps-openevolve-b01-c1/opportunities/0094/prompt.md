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
verified_results: {"accuracy": 0.9941, "parameters": 1536, "training_steps": 4999}
prior_hypothesis: Constraining `fc1` row 4 through its third input coordinate will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, showing that the failed row-5 constraint was another row-specific optimization failure rather than a limit on distributed quotients.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified `fc1` quotient from three rows to four will reduce the model from 1,543 to 1,542 parameters while retaining at least 99% accuracy, because each `fc1` row has the same redundant input-weight direction induced by the preceding bias-free LayerNorm.
change: Constrain the final input-weight coordinate of the first four `fc1` rows and canonicalize all four rows during fresh initialization, while retaining every LayerNorm scale and linear bias.
mechanism: Four-row pre-MLP LayerNorm/weight quotient
evidence_used: The same incremental quotient achieved 99.76% with one row, 99.90% with two rows, and 100% with three rows; this directly motivates testing one additional row without modifying any pathway implicated by prior failures.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1865, "parameters": 1542, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the last `fc1` row instead of the fourth row will produce a 1,542-parameter model with at least 99% accuracy, showing whether the previous collapse was specific to which initialized neuron received the fourth constraint.
change: Retain the successful constraints on rows 0–2, additionally remove the final input-weight coordinate from row 11, and canonicalize all four affected rows while preserving the freshly initialized function.
mechanism: Alternate-neuron fourth MLP weight quotient
evidence_used: Three consecutive constrained rows achieved 100% accuracy at 1,543 parameters, while adding row 3 collapsed to 18.65%; moving the fourth constraint to a different, otherwise exchangeable MLP neuron is the smallest test of a row-specific optimization failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1542, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the penultimate `fc1` row alongside rows 0–2 and 11 will reduce the model to 1,541 parameters while retaining at least 99% accuracy, indicating that the successful fourth constraint is characteristic of tail MLP neurons rather than unique to row 11.
change: Remove the final input-weight coordinate from `fc1` row 10 and canonicalize its fresh initialization, leaving all previously successful constraints and training settings unchanged.
mechanism: Tail-neuron fifth MLP weight quotient
evidence_used: Rows 0–2 achieved 100% accuracy at 1,543 parameters; adding row 3 collapsed to 18.65%, whereas constraining row 11 instead achieved 99.85% at 1,542 parameters. Testing adjacent row 10 is the most targeted next probe of this row-specific optimization effect.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1541, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 9 alongside rows 0–2 and 10–11 will reduce the model to 1,540 parameters while retaining at least 99% accuracy, indicating that the tail-neuron optimization tolerance extends beyond the final two rows.
change: Remove the final input-weight coordinate from `fc1` row 9 and canonicalize its fresh initialization, preserving all existing constraints and training settings.
mechanism: Tail-neuron sixth MLP weight quotient
evidence_used: Constraining rows 0–2 reached 100% at 1,543 parameters, row 3 caused collapse, while successively constraining tail rows 11 and 10 retained 99.85% and 99.96%; row 9 is the most informative next boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1540, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 8 alongside rows 0–2 and 9–11 will reduce the model to 1,539 parameters while retaining at least 99% accuracy, showing that the tail-neuron optimization tolerance extends through the final four rows.
change: Remove the final input-weight coordinate from `fc1` row 8 and canonicalize its fresh initialization, preserving all existing model and training pathways.
mechanism: Tail-neuron seventh MLP weight quotient
evidence_used: Constraints on rows 0–2 reached 100% at 1,543 parameters; row 3 caused collapse, while successively adding tail rows 11, 10, and 9 retained 99.85%, 99.96%, and 99.89%. Row 8 is the most informative next boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1539, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 7 alongside rows 0–2 and 8–11 will reduce the model from 1,539 to 1,538 parameters while retaining at least 99% accuracy, showing that the tail-neuron optimization tolerance extends through the final five rows.
change: Remove the final input-weight coordinate from `fc1` row 7 and canonicalize its fresh initialization, preserving all other model and training pathways.
mechanism: Tail-neuron eighth MLP weight quotient
evidence_used: Constraints on rows 0–2 reached 100% at 1,543 parameters; row 3 caused collapse, while incrementally constraining tail rows 11, 10, 9, and 8 retained 99.85%, 99.96%, 99.89%, and 100% accuracy. Row 7 is the most informative next boundary probe.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4022, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 6 while leaving the failed row 7 unconstrained will reduce the model from 1,539 to 1,538 parameters while retaining at least 99% accuracy, demonstrating another row-specific optimization effect.
change: Preserve constraints on rows 0–2 and 8–11, additionally remove row 6’s final input-weight coordinate, and canonicalize its fresh initialization without changing its initial map on zero-mean inputs.
mechanism: Alternate-neuron eighth MLP weight quotient
evidence_used: The seven-row model achieved 100% accuracy, whereas adding row 7 collapsed to 40.22%; earlier, moving a failed fourth constraint from row 3 to row 11 restored 99.85%, directly motivating another alternate-row probe.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6923, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining row 7 through its first input coordinate will reduce the model to 1,538 parameters while retaining at least 99% accuracy, because it avoids further concentrating all eight quotients on the final LayerNorm coordinate.
change: Keep rows 0–2 and 8–11 constrained on their final coordinate, but constrain row 7 on its first coordinate and canonicalize initialization to preserve its initial map on zero-mean inputs.
mechanism: Distributed-coordinate eighth MLP quotient
evidence_used: The 1,539-parameter seven-row model achieved 100%, while adding either row 7 or row 6 with the same final-coordinate constraint collapsed to 40.22% and 69.23%. This motivates testing whether the failure comes from excessive loss of one shared input coordinate rather than from the selected neuron alone.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1538, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing each head’s query coordinate system while preserving its independent learned attention-score function will reduce the model from 1,538 to 1,506 parameters and retain at least 99% accuracy.
change: Replace the redundant learned query/key coordinate frames with head-specific anchored query maps. Canonically transform fresh query initialization into complementary fixed coordinate blocks and compensate in each learned key map, preserving initial attention scores and all independent value pathways.
mechanism: Complementary head-anchored bilinear attention
evidence_used: Sharing one query projection across heads collapsed accuracy to 0.01%, showing that head-specific score functions are load-bearing. This patch retains separate query tails, keys, biases, and values for every head; it removes only the 16-parameter query/key basis gauge per head and preserves the freshly initialized score functions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2328, "parameters": 1506, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 6 through its second input coordinate will reduce the model from 1,538 to 1,537 parameters while retaining at least 99% accuracy, because it adds an exact LayerNorm-induced quotient without further concentrating constraints on either the final or first coordinate.
change: Keep the eight verified constraints, remove row 6’s second input weight, and canonicalize its fresh initialization to preserve its initial map on zero-mean inputs.
mechanism: Distributed-coordinate ninth MLP quotient
evidence_used: Eight constraints concentrated on the final coordinate failed for both row 7 and row 6, while moving row 7’s constraint to the first coordinate achieved 99.95%; distributing row 6’s new constraint onto another coordinate is the smallest targeted extension of that successful mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1537, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its third input coordinate will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, because it extends the successful coordinate-distribution pattern without concentrating another quotient on previously used coordinates.
change: Preserve the nine verified constraints, remove row 5’s third input weight, and canonicalize its fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.
mechanism: Staircase-distributed tenth MLP quotient
evidence_used: Adding row 6 on the second coordinate retained 99.95% after adding row 7 on the first coordinate also retained 99.95%, whereas concentrating additional constraints on the final coordinate caused severe collapses; row 5 on the third coordinate is the smallest targeted continuation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6651, "parameters": 1536, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 4 through its third input coordinate will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, showing that the failed row-5 constraint was another row-specific optimization failure rather than a limit on distributed quotients.
change: Leave row 5 unconstrained, remove row 4’s third input weight, and canonicalize its fresh initialization while preserving its initial map on zero-mean LayerNorm inputs.
mechanism: Alternate-neuron tenth MLP quotient
evidence_used: Moving the failed fourth final-coordinate constraint from row 3 to row 11 restored accuracy from 18.65% to 99.85%, establishing strong neuron-specific optimization effects. Since row 5 on the third coordinate reached only 66.51%, testing the adjacent exchangeable row on the same coordinate is the most targeted next probe.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9941, "parameters": 1536, "training_steps": 4999}



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
