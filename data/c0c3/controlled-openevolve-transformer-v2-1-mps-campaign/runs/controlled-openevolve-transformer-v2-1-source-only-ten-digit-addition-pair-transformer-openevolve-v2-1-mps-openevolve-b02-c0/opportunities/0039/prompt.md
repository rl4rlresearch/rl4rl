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
verified_results: {"accuracy": 0.9991, "parameters": 1628, "training_steps": 4999}
prior_hypothesis: Reusing the second learned positional scalar as the third-to-last row’s final coordinate will reduce the model from 1,629 to 1,628 parameters while retaining at least 99% accuracy, because it extends the successful randomly initialized learned positional tie to one adjacent row.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Anchoring one global scalar of the tied token embedding will reduce the model from 1,631 to 1,630 parameters while retaining at least 99% accuracy, because subtracting that scalar from every embedding coordinate produces only a LayerNorm-invariant input offset and a softmax-invariant common logit offset.
change: Store one fewer token-embedding scalar, reconstruct the final scalar as zero, initialize by subtracting the sampled final scalar globally, and preserve weight tying through a parameterless output projection wrapper.
mechanism: Global tied-token embedding common-mode gauge anchoring
evidence_used: A single positional common-mode anchor achieved 99.93% at 1,631 parameters, while extending it to a second positional row fell to 72.89%; this motivates testing the same exact normalization gauge in the distinct tied-token embedding rather than further constraining positional rows.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8468000000000001, "parameters": 1630, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one tied-token embedding scalar at zero while independently initializing every remaining scalar will reduce the model from 1,631 to 1,630 parameters and retain at least 99% accuracy by avoiding the global initialization correlation introduced by the previous gauge-preserving subtraction.
change: Store the tied token embedding with one fewer scalar, reconstruct only its final scalar as zero, initialize the compact learned coordinates normally, and use the reconstructed matrix through a parameterless tied output projection.
mechanism: Locally initialized tied-token common-mode gauge
evidence_used: The globally shifted token-embedding anchor reached only 84.68%, while a localized positional common-mode anchor reached 99.93%; this motivates retaining the exact token-embedding gauge reduction but localizing its initialization effect to one scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1630, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the penultimate positional row’s final coordinate at zero without shifting its other coordinates will reduce the model from 1,630 to 1,629 parameters while retaining at least 99% accuracy.
change: Compact the final two positional rows by one scalar each, preserve the successful final-row subtraction initialization, and independently initialize the penultimate row’s surviving coordinates.
mechanism: Locally initialized second positional-row gauge anchor
evidence_used: Subtracting an anchor from a second positional row failed at 72.89%, while replacing the failed globally shifted token anchor with a locally initialized fixed scalar recovered from 84.68% to 99.91%; this directly motivates testing the same localized initialization remedy for the second positional gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9645, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the first learned positional scalar as the penultimate row’s final coordinate will reduce the model from 1,630 to 1,629 parameters while retaining at least 99% accuracy, because that coordinate is a pure positional common-mode gauge but remains randomly initialized and dynamically learned.
change: Remove the penultimate positional row’s final scalar, reconstruct it from the first positional scalar, and preserve the successful final-row gauge initialization.
mechanism: Dynamic cross-position gauge tying
evidence_used: Fixing the second positional anchor locally reached 96.45%, while a learned LayerNorm tie reached 99.78% after its fixed counterpart reached only 37.17%; this motivates replacing the fixed second positional anchor with a learned tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1629, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the second learned positional scalar as the third-to-last row’s final coordinate will reduce the model from 1,629 to 1,628 parameters while retaining at least 99% accuracy, because it extends the successful randomly initialized learned positional tie to one adjacent row.
change: Remove the third-to-last positional row’s final scalar, reconstruct it from the second positional scalar, and retain the existing penultimate learned tie and final-row zero anchor.
mechanism: Incremental dynamic cross-position gauge tying
evidence_used: Replacing the penultimate row’s fixed-zero coordinate, which achieved only 96.45%, with a learned cross-position tie achieved 99.91% at 1,629 parameters; this directly motivates one incremental tie using a distinct learned scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1628, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the third learned positional scalar as the fourth-to-last row’s final coordinate will reduce the model from 1,628 to 1,627 parameters while retaining at least 99% accuracy.
change: Remove the fourth-to-last positional row’s final scalar, reconstruct it from the third positional scalar, and retain the two successful learned ties and final-row zero anchor.
mechanism: Incremental dynamic cross-position gauge tying
evidence_used: The first learned positional tie achieved 99.91% at 1,629 parameters, and extending it to an adjacent row with a distinct learned scalar again achieved 99.91% at 1,628; this directly motivates one further incremental tie using another distinct learned scalar.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5313, "parameters": 1627, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the third learned positional scalar for the fourth-to-last row while blocking that row’s reverse gradient will reduce the model to 1,627 parameters and retain at least 99% accuracy by avoiding the cross-position gradient interference implicated by the prior 53.13% result.
change: Remove the fourth-to-last positional row’s final scalar, reconstruct it from a detached view of the third learned positional scalar, and retain the two successful learned ties and final-row zero anchor.
mechanism: Stop-gradient cross-position gauge tying
evidence_used: Two dynamic positional ties retained 99.91% accuracy at 1,629 and 1,628 parameters, but adding a third tie collapsed to 53.13%; isolating the new tie’s backward coupling tests whether optimization interference, rather than the gauge reduction itself, caused that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.27690000000000003, "parameters": 1627, "training_steps": 4999}



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
