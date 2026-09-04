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
verified_results: {"accuracy": 1.0, "parameters": 1609, "training_steps": 4999}
prior_hypothesis: Learning two of eight query-bias coordinates will reduce the model from 1610 to 1609 parameters while retaining at least 99% accuracy, because every prior query-bias reduction through three learned coordinates retained at least 99.98% accuracy while preserving the necessary three-coordinate value bias.

## Recent verification evidence

RECENT RESULT
hypothesis: Representing each learned position with seven coordinates and padding an eighth zero will remove `max_seq_len` parameters while retaining at least 99% accuracy, because position-dependent shifts along the all-ones direction are invisible to every pre-LayerNorm sublayer and the final LayerNorm.
change: Decouple positional-code width from the load-bearing eight-dimensional residual stream by using a seven-dimensional learned positional embedding with a fixed zero gauge coordinate.
mechanism: LayerNorm-gauge-fixed positional codes
evidence_used: Reducing the entire residual width to six collapsed accuracy to 0.01%, while the eight-wide model reaches 99.91%. Prior designs assumed positional and computational widths must match; this patch preserves all attention and MLP capacity while removing a functionally redundant positional degree of freedom.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.005600000000000001, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the learned value bias from five to four coordinates will lower parameters from 1617 to 1616 while retaining at least 99% accuracy, because the three-coordinate ablation achieved 99.91% and remains far above the 97.53% observed only when all eight coordinates were removed.
change: Store four learned value-bias coordinates and pad the remaining four coordinates with zeros during attention.
mechanism: Four-coordinate value-bias ablation
evidence_used: The current five-coordinate value bias reached 99.91% with 1617 parameters, while six coordinates reached 99.58%; continuing the one-coordinate ablation is the most direct test of the accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1616, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the learned value bias from four to three coordinates will lower parameters from 1616 to 1615 while retaining at least 99% accuracy, because the four-coordinate version achieved 99.98% and three learned coordinates may preserve enough of the optimization benefit lost only when all eight coordinates were removed.
change: Store three learned value-bias coordinates and pad the remaining five coordinates with zeros during attention.
mechanism: Five-coordinate value-bias ablation
evidence_used: The current four-coordinate value bias reached 99.98% with 1616 parameters, while removing all eight coordinates reached 97.53%; another one-coordinate ablation is the most informative test of the parameter boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the learned value bias from three coordinates to two will lower parameters from 1615 to 1614 while retaining at least 99% accuracy, because the three-coordinate design achieved 100% accuracy and two coordinates may preserve enough of the optimization benefit lost only when all eight were removed.
change: Store two learned value-bias coordinates and pad the remaining six coordinates with zeros during attention.
mechanism: Six-coordinate value-bias ablation
evidence_used: The current three-coordinate value bias reached 100% accuracy with 1615 parameters, while removing all eight coordinates reached 97.53%; another one-coordinate ablation is the most informative test of the parameter boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5514, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning seven of eight query-bias coordinates will reduce the model from 1615 to 1614 parameters while retaining at least 99% accuracy, because it preserves the three-coordinate value bias that achieved 100% accuracy and makes only a one-coordinate change to the otherwise verified architecture.
change: Store seven learned query-bias coordinates and pad the remaining coordinate with zero during attention.
mechanism: Single-coordinate query-bias ablation
evidence_used: Reducing the value bias from three coordinates to two collapsed accuracy from 100% to 55.14%, identifying the three-coordinate value bias as a boundary worth preserving; a one-coordinate ablation of the still-intact eight-coordinate query bias is therefore the most informative alternative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning six of eight query-bias coordinates will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because the seven-coordinate query bias achieved 100% accuracy while preserving the verified three-coordinate value bias.
change: Store six learned query-bias coordinates and pad the remaining two coordinates with zeros during attention.
mechanism: Second-coordinate query-bias ablation
evidence_used: The single-coordinate query-bias ablation reduced parameters to 1614 with 100% accuracy, whereas reducing the three-coordinate value bias caused accuracy to collapse to 55.14%; continuing the query-bias ablation is therefore the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1613, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning five of eight query-bias coordinates will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy, because the six-coordinate query bias achieved 100% accuracy while preserving the necessary three-coordinate value bias.
change: Store five learned query-bias coordinates and pad the remaining three coordinates with zeros during attention.
mechanism: Third-coordinate query-bias ablation
evidence_used: The six-coordinate query-bias design reached 100% accuracy with 1613 parameters, while reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the successful query-bias ablation is the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1612, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning four of eight query-bias coordinates will reduce the model from 1612 to 1611 parameters while retaining at least 99% accuracy, because the five- and six-coordinate query-bias designs both achieved 100% while preserving the necessary three-coordinate value bias.
change: Store four learned query-bias coordinates and pad the remaining four coordinates with zeros during attention.
mechanism: Fourth-coordinate query-bias ablation
evidence_used: Successive query-bias reductions from eight to seven, six, and five learned coordinates all retained 100% accuracy, whereas reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the query-bias ablation is the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning three of eight query-bias coordinates will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy, because the four-coordinate query-bias design achieved 99.99% while preserving the necessary three-coordinate value bias.
change: Store three learned query-bias coordinates and pad the remaining five coordinates with zeros during attention.
mechanism: Fifth-coordinate query-bias ablation
evidence_used: Successive query-bias reductions to seven, six, five, and four learned coordinates achieved at least 99.99% accuracy, while reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the successful query-bias ablation is the most informative next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1610, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning two of eight query-bias coordinates will reduce the model from 1610 to 1609 parameters while retaining at least 99% accuracy, because every prior query-bias reduction through three learned coordinates retained at least 99.98% accuracy while preserving the necessary three-coordinate value bias.
change: Store two learned query-bias coordinates and pad the remaining six coordinates with zeros during attention.
mechanism: Sixth-coordinate query-bias ablation
evidence_used: The three-coordinate query-bias design achieved 99.98% accuracy with 1610 parameters, and successive reductions from eight through three coordinates all exceeded 99.99% except the latest still-strong result; continuing this ablation is the most informative test of its capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1609, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the bias-free second head’s key and value projections will reduce the model from 1609 to 1577 parameters while retaining at least 99% accuracy, because the first head retains fully independent projections and all three load-bearing value-bias coordinates.
change: Challenge the assumption that every attention head needs separate key and value maps by giving the second head a shared learned associative representation, while preserving the fully general bias-carrying first head.
mechanism: Bias-localized key–value sharing
evidence_used: The current model achieved 100% after removing every query- and value-bias coordinate from the second head, whereas reducing the first head’s value bias from three coordinates to two collapsed accuracy to 55.14%; this identifies the first head as the safer place to preserve capacity and the bias-free second head as the informative target for sharing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5229, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Learning one of eight query-bias coordinates will reduce the model from 1609 to 1608 parameters while retaining at least 99% accuracy, because every prior query-bias reduction through two learned coordinates retained at least 99.98% accuracy while preserving the necessary three-coordinate value bias.
change: Store one learned query-bias coordinate and pad the remaining seven coordinates with zeros during attention.
mechanism: Seventh-coordinate query-bias ablation
evidence_used: The two-coordinate query-bias design achieved 100% accuracy with 1609 parameters, while reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the successful query-bias ablation is the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8417, "parameters": 1608, "training_steps": 4999}



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
