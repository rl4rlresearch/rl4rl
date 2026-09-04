# Improve fixed-exposure image classification

You are an autonomous ML engineer improving a learned classifier for 28×28
grayscale images in ten classes.

## Goal

Maximize `validation_score`. It ranks implementations first by the exact number
of correct predictions on the fixed 10,000-image validation set, then—only when
correct counts tie—by lower validation cross-entropy. Every verification starts
from a fresh initialization and presents exactly 100,000 examples from the
fixed 50,000-image training split.

You may change the model architecture, optimizer, loss, augmentation, batch
size, gradient handling, schedule, and other contents of `train.py`. The fixed
data split, normalization, example accounting, validation calculation,
250,000-learned-parameter ceiling, and device are not editable. The protected
loop calls the functions already defined in `train.py`; keep that interface
intact. The model must return one ten-class logit vector per image.

## Work boundaries

Maximize validation_score. No additional accuracy threshold.
Editable source files: train.py.
Results reported after each verification: validation_score, validation_correct, validation_accuracy, validation_cross_entropy, parameters, examples_processed, optimizer_steps, training_seconds, batch_size.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, external datasets, pretrained weights, or any
surrounding repository. Do not run training or validation yourself and do not
generate hidden alternatives. Return one patch for one implementation;
verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}
prior_hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.72801075014286, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.22454801788330078, "validation_score": 9208.408313918848}
prior_hypothesis: Halving the two unshifted-view weights will exceed 9,206 correct predictions by moving aggregation opposite the clean-view doubling that reduced correctness to 9,202.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 69.44678320805542, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.225365966796875, "validation_score": 9209.4080413636}
prior_hypothesis: Replacing one horizontally redundant shift pair with centered views while preserving the best verified 1:8 centered-to-shifted weight ratio will reach at least 9,208 correct predictions without increasing inference work.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.40000991616398, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22445868225097657, "validation_score": 9206.408343709141}
prior_hypothesis: The verified per-view sharpening and 0.0325 BatchNorm EMA will recover 9,206 correct predictions, while increasing the argmax-invariant outer sharpening to 1.10× will reduce cross-entropy below 0.227083.

## Recent verification evidence

RECENT RESULT
hypothesis: Excluding both unshifted views will exceed 9,208 correct predictions by continuing the observed improvement as their weights decreased from 2.0 to 1.0 to 0.5, while reducing inference work.
change: Restore the qualified residual architecture, phase-specific parameter-and-BatchNorm EMA, and calibrated sharpening, then ensemble only the eight ±1-pixel shifted and flipped views.
mechanism: Shift-only calibrated residual ensemble
evidence_used: Unshifted weights 2.0, 1.0, and 0.5 produced 9,202, 9,206, and 9,208 correct respectively. The two 0.25 attempts timed out without contrary accuracy evidence, motivating a computationally cheaper test of the zero-weight endpoint.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 73.22419091686606, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22472651176452638, "validation_score": 9206.408254410431}

RECENT RESULT
hypothesis: Reducing each unshifted-view weight from 0.5 to 0.375 will exceed 9,208 correct predictions by locating a better interior balance between the inferior zero-weight and equal-weight endpoints.
change: Give the original and flipped unshifted predictions 0.375 weight each while retaining unit-weight shifted views, then normalize by total weight 8.75.
mechanism: Fine-grained interior TTA weight interpolation
evidence_used: Unshifted weights 0.0, 0.5, and 1.0 produced 9,206, 9,208, and 9,206 correct respectively, demonstrating an interior optimum; the unresolved 0.25 attempts supplied no contrary performance result.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding lightweight channel gates while retaining the qualified half-weight unshifted TTA will exceed 9,208 correct predictions by adapting feature-channel importance to each image without the runtime cost that prevented learned mixed pooling from finishing.
change: Add squeeze-excitation gating to every residual stage, shrink the dense hidden layer from 48 to 44 to remain below the parameter ceiling, and use the best verified 0.5 unshifted-view weighting.
mechanism: Input-conditioned squeeze-excitation residual channels
evidence_used: Reference Design 1 achieved the best result, 9,208 correct, through TTA weighting alone; subsequent aggregation changes did not progress, while learned mixed pooling timed out. This challenges the shared assumption of sample-independent channel importance using a much cheaper dynamic representation mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the qualified 0.5-weight centered views will recover 9,208 correct predictions, while increasing post-ensemble sharpening from 1.10× to 1.15× will preserve those predictions and reduce cross-entropy below 0.224548.
change: Add the original and horizontally flipped views at half weight alongside the eight shifted views, normalize by 9.0, and apply 1.15× final sharpening.
mechanism: Half-weight centered TTA with stronger argmax-invariant calibration
evidence_used: Reference Design 1 verified half-weight centered views at 9,208 correct and 0.224548 cross-entropy, outperforming the current shift-only design’s 9,206 correct. Reference Designs 2 and 3 showed that increasing final sharpening from 1.05× to 1.10× preserved correctness while lowering cross-entropy from 0.227083 to 0.224459.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Matching terminal augmentation frequencies to the best verified half-weight centered TTA will exceed 9,208 correct predictions by training on centered versus shifted views in the same 1:8 ratio used during validation.
change: Restore the qualified half-weight centered ensemble with 1.10× calibration, and sample terminal crops with one centered outcome and two outcomes for each cardinal shift.
mechanism: TTA-matched terminal shift curriculum
evidence_used: Reference Design 1 achieved 9,208 correct with total centered weight 1.0 and shifted weight 8.0, outperforming equal weighting and shift-only aggregation; the current uniform terminal sampler instead presents centered examples at a mismatched 1:4 ratio.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 71.57632437511347, "validation_accuracy": 0.9193, "validation_correct": 9193, "validation_cross_entropy": 0.2254965400695801, "validation_score": 9193.407997887918}

RECENT RESULT
hypothesis: Increasing final sharpening from 1.10× to 1.15× will preserve the current 9,208 predictions while reducing validation cross-entropy below 0.224548.
change: Increase only the argmax-invariant multiplier applied after the qualified half-weight centered-view ensemble.
mechanism: Post-ensemble temperature sharpening
evidence_used: Raising final sharpening from 1.05× to 1.10× preserved 9,206 correct while reducing cross-entropy from 0.227083 to 0.224459; the prior 1.15× attempt timed out and supplied no contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Half-weighting both centered views will reproduce the qualified 9,208-correct result, exceeding the current 9,206 correct predictions.
change: Weight the original and flipped centered predictions by 0.5 while retaining unit-weight shifted views, then normalize the ensemble by 9.0.
mechanism: Shift-view-favored probability TTA
evidence_used: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy, outperforming the current equal-weight aggregation at 9,206 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the verified half-weight centered ensemble will recover 9,208 correct predictions, while a 1.125× final multiplier will preserve its argmax decisions and reduce cross-entropy below 0.224548.
change: Add centered and horizontally flipped predictions at 0.5 weight, retain all eight unit-weight shifted predictions, normalize by 9.0, and sharpen the resulting logits by 1.125×.
mechanism: Half-weight centered TTA with conservative temperature refinement
evidence_used: Reference Design 1 achieved the best verified correctness, 9,208, with half-weight centered views; increasing final sharpening from 1.05× to 1.10× previously preserved correctness and reduced cross-entropy, motivating a conservative midpoint toward the unresolved 1.15× test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The qualified half-weight centered ensemble will reproduce 9,208 correct predictions, exceeding the current 9,206, while 1.10× final sharpening reduces cross-entropy.
change: Give each centered view 0.5 weight, retain unit-weight shifted views, normalize by 9.0, and increase final sharpening from 1.05× to 1.10×.
mechanism: Half-weight centered-view calibrated probability TTA
evidence_used: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy; later verification failures supplied no contrary performance result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A centered-view weight of 0.375 will exceed 9,208 correct predictions by refining the observed interior optimum between the inferior zero- and unit-weight endpoints, while paired inference will avoid the prior verification timeout.
change: Batch each view with its horizontal flip in one forward pass, reduce both centered-view weights from 0.5 to 0.375, and normalize the unchanged eight unit-weight shifted views by the resulting total weight of 8.75.
mechanism: Pair-batched interior-weight probability TTA
evidence_used: Centered weights 0.0, 0.5, and 1.0 produced 9,206, 9,208, and 9,206 correct respectively; the previous 0.375 test timed out without performance evidence, so compute-equivalent paired inference makes that unresolved interpolation test more likely to complete.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Half-weighting both centered views will reproduce the qualified 9,208-correct result, exceeding the current 9,206 correct predictions.
change: Give each centered view 0.5 weight, retain unit weight for the eight shifted views, and normalize by total weight 9.0.
mechanism: Shift-favored probability TTA
evidence_used: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy; subsequent verification failures reported no contrary accuracy result, while the zero- and unit-weight endpoints each achieved only 9,206 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing one horizontally redundant shift pair with centered views while preserving the best verified 1:8 centered-to-shifted weight ratio will reach at least 9,208 correct predictions without increasing inference work.
change: Evaluate centered and flipped-centered views at 0.375 weight each, retain three unit-weight shift pairs, omit one horizontal shift pair, and normalize the eight-pass ensemble by 6.75.
mechanism: Flip-symmetric eight-pass centered TTA
evidence_used: Half-weight centered views with eight shifted views achieved the best verified result of 9,208 correct at a 1:8 centered-to-shifted weight ratio, whereas shift-only inference achieved 9,206. Prior ten-view attempts repeatedly failed verification, motivating the same ratio within the current eight-forward runtime budget.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 69.44678320805542, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.225365966796875, "validation_score": 9209.4080413636}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
blocks have been applied. All blocks must apply. Together they must describe
one implementation ready for verification. The mechanism name is descriptive,
not chosen from a fixed list. Do not paste whole files, lengthy logs, or routine
progress reports outside the patch.
