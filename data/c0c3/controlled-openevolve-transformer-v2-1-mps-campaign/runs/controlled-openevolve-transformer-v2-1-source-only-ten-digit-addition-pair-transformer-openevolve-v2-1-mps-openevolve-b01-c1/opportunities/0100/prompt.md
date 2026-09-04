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
verified_results: {"accuracy": 0.9987, "parameters": 1535, "training_steps": 4999}
prior_hypothesis: Constraining `fc1` row 3 through its fourth input coordinate will reduce the model from 1,536 to 1,535 parameters while retaining at least 99% accuracy, showing that row 5’s failure on this coordinate was neuron-specific.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its fourth input coordinate will reduce the model from 1,536 to 1,535 parameters while retaining at least 99% accuracy, because changing the constrained coordinate previously rescued row 7 and avoids reusing row 5’s failed third-coordinate constraint.
change: Remove row 5’s fourth input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve the row’s initial map on zero-mean LayerNorm inputs.
mechanism: Coordinate-alternate eleventh MLP quotient
evidence_used: Row 5 constrained on the third coordinate collapsed to 66.51%, while row 7 improved from 40.22% with a final-coordinate constraint to 99.95% with a first-coordinate constraint. This directly motivates testing whether a different distributed coordinate similarly rescues row 5.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.379, "parameters": 1535, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 3 through its fourth input coordinate will reduce the model from 1,536 to 1,535 parameters while retaining at least 99% accuracy, showing that row 5’s failure on this coordinate was neuron-specific.
change: Remove row 3’s fourth input weight, reconstruct it as zero during the forward pass, and canonicalize its fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.
mechanism: Alternate-neuron staircase MLP quotient
evidence_used: Row 5 constrained on its third coordinate failed at 66.51%, while moving that same constraint to row 4 achieved 99.41%; row 5’s fourth-coordinate failure at 37.9% therefore most directly motivates testing the same coordinate on the remaining unconstrained row 3.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1535, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its fifth input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because coordinate choice has shown strong neuron-specific effects and this coordinate remains untested for row 5.
change: Remove row 5’s fifth input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve the row’s initial map on zero-mean LayerNorm inputs.
mechanism: Coordinate-alternate twelfth MLP quotient
evidence_used: Row 5 failed when constrained on its third and fourth coordinates, but moving row 7’s constraint from its final coordinate to its first rescued accuracy from 40.22% to 99.95%. This makes another coordinate-specific row-5 probe the smallest informative extension of the 1,535-parameter design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0993, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining `fc1` row 5 through its first input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because the first-coordinate parameterization previously rescued row 7 and avoids row 5’s unsuccessful third-, fourth-, and fifth-coordinate choices.
change: Remove row 5’s first input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.
mechanism: First-coordinate twelfth MLP quotient
evidence_used: Row 5 achieved only 66.51%, 37.9%, and 9.93% when constrained through coordinates three, four, and five, while moving row 7’s constraint to the first coordinate raised accuracy from 40.22% to 99.95%; this makes the first coordinate the strongest untested row-5 alternative.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3961, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final `ln2` scale at one will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because its effect remains representable by the learned final input column of `fc1` and fresh initialization is unchanged.
change: Remove the final trainable scale coordinate from the bias-free MLP LayerNorm and reconstruct it as a fixed one during the forward pass.
mechanism: Final-scale/downstream-column gauge fixing
evidence_used: The 1,535-parameter design achieved 99.87% accuracy with seven `fc1` rows already independent of the final input coordinate, while four attempts to constrain the remaining row 5 failed. This motivates preserving row 5 and instead testing the exact upstream-scale/downstream-column gauge, which affects only the five rows retaining that column.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7492, "parameters": 1534, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-weight coordinate through a score-preserving orthogonal query/key rotation will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, avoiding the optimization shock caused by fixing all 32 attention-basis gauges simultaneously.
change: Replace the QKV projection with a one-parameter-smaller projection whose first query coordinate is fixed at zero, and orthogonally canonicalize the corresponding query/key rows during fresh initialization while preserving initial attention scores.
mechanism: Single-coordinate orthogonal query/key gauge fixing
evidence_used: The 1,535-parameter design reached 99.87%, whereas removing all 32 query/key basis-gauge parameters at once collapsed to 23.28% despite preserving score functions. Fixing only one gauge coordinate is the smallest test of whether that failure resulted from aggressive simultaneous reparameterization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7392, "parameters": 1534, "training_steps": 4999}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
