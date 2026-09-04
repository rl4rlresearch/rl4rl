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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1631, "training_steps": 4999}
prior_hypothesis: Anchoring one coordinate of the final positional embedding row at zero will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because a position-specific all-coordinate offset propagates only through residual streams and is removed by every downstream LayerNorm.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing one `ln2` bias coordinate at zero will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because the omitted offset can be absorbed exactly into the following `fc1` bias.
change: Replace the standard pre-MLP LayerNorm with the existing initialization-preserving `AnchoredLayerNorm`, which learns seven bias coordinates and appends one fixed zero.
mechanism: Pre-MLP LayerNorm-to-linear bias gauge anchoring
evidence_used: A one-coordinate LayerNorm bias anchor previously achieved 99.95% accuracy, and the current design also achieves 99.95%; applying the same successful reduction before `fc1` preserves initialization and removes an exact LayerNorm-linear parameter redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1353, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the last `ln1` scale coordinate at one will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because its matching bias coordinate is already fixed at zero and any learned scale can be absorbed exactly into the corresponding QKV weight column.
change: Store seven learned `ln1` scale coordinates and append one fixed-one coordinate during the forward pass, preserving the initial function exactly.
mechanism: Joint LayerNorm shift-and-scale gauge anchoring
evidence_used: Anchoring the last `ln1` bias coordinate previously achieved 99.95% accuracy; extending that successful parameterization at the same coordinate removes an exact LayerNorm-to-linear scale redundancy without altering initialization, unlike the failed final-MLP weight anchor.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.37170000000000003, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the shared key-bias scalar to an existing learned query-bias coordinate will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because the key-bias contribution remains position-independent and softmax-invariant while preserving the successful dynamically learned, head-shared key-bias pathway.
change: Store only the eight query and eight value bias coordinates, and reuse the first learned query-bias scalar as the fourth key-bias coordinate in both attention heads.
mechanism: Cross-role query/key bias tying
evidence_used: The learned head-shared key-bias design reached 99.92%, whereas replacing that scalar with fixed zeros collapsed accuracy to 45.19%; tying it to an existing learned coordinate removes one parameter while retaining a varying shared key-bias signal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9932, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the last `ln1` scale coordinate to the first will reduce the model from 1,634 to 1,633 parameters while retaining at least 99% accuracy, because the following QKV weight columns can absorb independent scale differences while the tied coordinate remains dynamically learned.
change: Store seven `ln1` scale parameters and reuse the first learned scale as the eighth coordinate, preserving the initial all-ones scale exactly.
mechanism: Dynamic LayerNorm scale tying
evidence_used: Fixing the last `ln1` scale at one failed at 37.17%, while dynamically tying the otherwise invariant key-bias scalar to a query-bias coordinate recovered 99.32% at 1,634 parameters; this directly motivates replacing a fixed scale anchor with a learned tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9978, "parameters": 1633, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the second-to-last `ln1` scale coordinate to the second learned coordinate will reduce the model from 1,633 to 1,632 parameters while retaining at least 99% accuracy, because the following QKV weight columns can absorb independent scale differences while all tied coordinates remain dynamically learned.
change: Store six learned `ln1` scale parameters, reconstruct the seventh from the second and the eighth from the first, and preserve the initial all-ones scale exactly.
mechanism: Incremental dynamic LayerNorm scale tying
evidence_used: Dynamically tying the eighth `ln1` scale to the first achieved 99.78% accuracy at 1,633 parameters, whereas fixing it at one achieved only 37.17%; this supports one further incremental learned tie rather than a fixed anchor.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6869, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the last value-bias coordinate to the first will reduce the model from 1,633 to 1,632 parameters while retaining at least 99% accuracy, because attention-weighted value bias is a learned residual-stream offset that the unrestricted output projection bias can absorb.
change: Store seven value-bias coordinates and reuse the first as the eighth, while preserving the successful query-tied key-bias layout and the initial all-zero function.
mechanism: Dynamic value-bias tying
evidence_used: Dynamic tying succeeded for the first `ln1` scale reduction at 99.78% and for the query/key bias reduction at 99.32%, whereas a second `ln1` scale tie failed; this motivates testing a single learned tie in the distinct, projection-redundant value-bias pathway.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the second-to-last value-bias coordinate to the second learned value coordinate will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because attention-weighted value bias is a constant learned offset that the unrestricted output projection bias can absorb.
change: Store six value-bias coordinates, reconstruct the seventh from the second and the eighth from the first, while preserving the successful query-tied key-bias and initial all-zero function.
mechanism: Incremental dynamic value-bias tying
evidence_used: The first dynamic value-bias tie achieved 99.84% accuracy at 1,632 parameters; this motivates one incremental nested tie in the same projection-redundant pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9537, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the last `ln2` scale coordinate to its first learned coordinate will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because `fc1` can absorb independent scale differences while the tied scale remains dynamically learned.
change: Add a scale-tied LayerNorm that preserves all eight bias coordinates but learns seven scale coordinates, and use it for `ln2`.
mechanism: Independent pre-MLP dynamic LayerNorm scale tying
evidence_used: A single dynamic `ln1` scale tie achieved 99.78% accuracy, whereas a second simultaneous scale tie failed and fixing an `ln2` bias coordinate achieved only 13.53%; this motivates one learned scale tie in the distinct pre-MLP normalization while leaving every `ln2` bias trainable.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.682, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing an attention output-projection bias coordinate as the second-to-last value-bias coordinate will reduce the model to 1,631 parameters while retaining at least 99% accuracy, because value bias is converted into a constant residual offset that the output-projection bias already parameterizes.
change: Store six independent value-bias coordinates, retain the successful first-to-last value-bias tie, and reconstruct the seventh coordinate from the learned attention projection bias.
mechanism: Cross-module value/output-bias tying
evidence_used: The first dynamic value-bias tie achieved 99.84% accuracy at 1,632 parameters, while tying a second pair within the value bias reached only 95.37%; cross-module tying tests whether preserving a distinct learned signal from the projection-bias pathway avoids that optimization loss.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7326, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the second-to-last `ln1` bias coordinate to the first learned coordinate will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because the following biased QKV projection can absorb independent LayerNorm offsets while the tied coordinate remains dynamically learned.
change: Store six learned `ln1` bias coordinates, reconstruct the seventh from the first, and retain the fixed-zero eighth coordinate without changing the all-zero initialization.
mechanism: Incremental dynamic pre-attention LayerNorm bias tying
evidence_used: The existing one-coordinate `ln1` bias anchor previously achieved 99.95% accuracy, and a learned `ln1` scale tie achieved 99.78%; unlike the failed second scale tie, this edit targets the additive LayerNorm-to-QKV bias redundancy and preserves a trainable signal.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9752, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring one coordinate of the final positional embedding row at zero will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because a position-specific all-coordinate offset propagates only through residual streams and is removed by every downstream LayerNorm.
change: Replace the positional embedding with an initialization-preserving embedding that stores one fewer scalar and reconstructs the final coordinate as zero.
mechanism: Single-position embedding common-mode gauge anchoring
evidence_used: The current design reaches 99.84% at 1,632 parameters, and final-MLP common-mode bias anchoring previously retained 99.95%; this tests the same normalization gauge at a single positional row, minimizing the optimization disturbance that likely caused the attention-projection anchor failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the final coordinate of the penultimate positional embedding row will reduce the model from 1,631 to 1,630 parameters while retaining at least 99% accuracy, because subtracting a row-specific all-coordinate offset is removed by every downstream LayerNorm and the same anchoring on the final row achieved 99.93%.
change: Store both final positional rows without their last coordinate, reconstruct those coordinates as zero, and initialize each row by subtracting its sampled final coordinate.
mechanism: Incremental positional-row common-mode gauge anchoring
evidence_used: Single-position embedding common-mode anchoring achieved 99.93% accuracy at 1,631 parameters, strongly motivating one incremental application of the identical gauge to the adjacent positional row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7289, "parameters": 1630, "training_steps": 4999}



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
