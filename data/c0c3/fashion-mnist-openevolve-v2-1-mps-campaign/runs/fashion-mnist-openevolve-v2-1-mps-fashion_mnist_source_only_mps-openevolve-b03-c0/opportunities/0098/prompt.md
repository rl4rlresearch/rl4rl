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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 66.09472049982287, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.19801389999389649, "validation_score": 9321.417357427992}
prior_hypothesis: Allowing the 30% blend to change predictions only when all four translated views agree with its alternative class will exceed 9,319 correct predictions.

## Recent verification evidence

RECENT RESULT
hypothesis: A 27% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198298140.
change: Increase the evaluation-only translated-logit blend from 25% to 27%, retaining argmax gating and temperature calibration.
mechanism: Argmax-preserving 27% translation blend
evidence_used: Increasing the gated blend from 20% to 25% preserved 9,319 correct predictions and reduced cross-entropy from 0.198478033 to 0.198298140; 27% is a conservative step toward the fitted optimum near 28.55%, while the prior 28% timeout supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 70.15812437492423, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1981851608276367, "validation_score": 9319.417297773622}

RECENT RESULT
hypothesis: A 28% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198185161.
change: Increase the evaluation-only translated-logit blend from 27% to 28%, retaining argmax gating and temperature calibration.
mechanism: Near-optimal argmax-preserving translation blend
evidence_used: The 27% blend preserved 9,319 correct predictions and improved cross-entropy from 0.198298140 at 25% to 0.198185161; 28% moves toward the evidence-derived optimum near 28.55%, and its prior timeout provided no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 72.18467708397657, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1981589729309082, "validation_score": 9319.417306894407}

RECENT RESULT
hypothesis: A 29% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198158973.
change: Increase the evaluation-only translated-logit blend from 28% to 29%, retaining argmax gating and temperature calibration.
mechanism: Locally optimized argmax-preserving translation blend
evidence_used: Cross-entropy improved from 0.198298140 at 25% to 0.198185161 at 27% and 0.198158973 at 28%; a quadratic fit to these recent points places the local minimum near 28.8%, while the gate prevents prediction changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 66.88116679200903, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1981161933898926, "validation_score": 9319.41732179463}

RECENT RESULT
hypothesis: A 29.5% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198116193.
change: Increase the evaluation-only translated-logit blend from 29% to 29.5%, retaining argmax gating and temperature calibration.
mechanism: Argmax-preserving 29.5% translation blend
evidence_used: The 29% blend preserved 9,319 correct predictions and improved cross-entropy from 0.198158973 at 28% to 0.198116193; a half-point increase probes the nearby optimum more precisely, while the gate prevents prediction changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 59.459541792050004, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19809767379760743, "validation_score": 9319.41732824538}

RECENT RESULT
hypothesis: Increasing the gated translation contribution from 29.5% to 30% will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198097674.
change: Increase the evaluation-only translated-logit blend from 29.5% to 30%, retaining argmax gating and temperature calibration.
mechanism: Argmax-preserving 30% translation blend
evidence_used: The 29.5% blend preserved 9,319 correct predictions and improved cross-entropy from 0.198116193 at 29% to 0.198097674; the prior 30% attempt timed out and therefore supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 65.53753062500618, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19808012237548828, "validation_score": 9319.417334359081}

RECENT RESULT
hypothesis: Increasing the gated translation contribution from 30% to 30.5% will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198080122.
change: Increase the evaluation-only translated-logit blend from 30% to 30.5%, retaining argmax gating and temperature calibration.
mechanism: Argmax-preserving 30.5% translation blend
evidence_used: Cross-entropy improved monotonically from 0.198116193 at 29% to 0.198097674 at 29.5% and 0.198080122 at 30%; another half-point increment tests whether that local improvement continues while the gate prevents prediction changes.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 68.68824129202403, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.198105948638916, "validation_score": 9319.41732536306}

RECENT RESULT
hypothesis: A 29.95% translation contribution will preserve all 9,319 predictions and reduce validation cross-entropy below 0.198080122.
change: Refine the evaluation-only blend from 30% to 29.95%, retaining argmax gating and temperature calibration.
mechanism: Quadratic-refined argmax-preserving translation blend
evidence_used: Cross-entropy was 0.198097674 at 29.5%, improved to 0.198080122 at 30%, then worsened to 0.198105949 at 30.5%; a quadratic fit places the local minimum near 29.95%.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 81.34147662506439, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19808134155273438, "validation_score": 9319.417333934398}

RECENT RESULT
hypothesis: A 30.06% translation contribution will preserve all 9,319 predictions and reduce validation cross-entropy below 0.198080122.
change: Refine the evaluation-only blend from 30% to 30.06%, retaining argmax gating and temperature calibration.
mechanism: Locally interpolated argmax-preserving translation blend
evidence_used: Cross-entropy was 0.198081342 at 29.95%, improved to 0.198080122 at 30%, and worsened to 0.198105949 at 30.5%; local quadratic interpolation of these verified points places the minimum near 30.06%.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 78.24455475015566, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19811260147094725, "validation_score": 9319.417323045753}

RECENT RESULT
hypothesis: Translating 25% of training images by one reflected pixel will improve spatial robustness and produce more than 9,319 correct validation predictions.
change: Randomly replace one quarter of each training batch with a uniformly selected one-pixel translation matching the four evaluation-time views.
mechanism: Sparse one-pixel translation augmentation
evidence_used: Evaluation-time translated logits consistently reduced cross-entropy while preserving all 9,319 predictions; with the blend now locally saturated at 30%, training on those same perturbations is the most direct test of whether their complementary signal can improve correctness.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Allowing the 30% blend to change predictions only when all four translated views agree with its alternative class will exceed 9,319 correct predictions.
change: Retain the existing blend for argmax-preserving cases, while accepting prediction changes supported unanimously by all four translated views.
mechanism: Consensus-gated translation correction
evidence_used: Translation blending consistently reduced cross-entropy, indicating complementary signal; the current argmax gate prevents testing whether its strongest, view-consistent corrections improve accuracy.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 66.09472049982287, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.19801389999389649, "validation_score": 9321.417357427992}

RECENT RESULT
hypothesis: Allowing the 30% blend to change a base prediction when at least three translated views support the alternative will exceed 9,321 correct validation predictions.
change: Relax the evaluation-time correction gate from unanimous four-view agreement to agreement by at least three of four translated views.
mechanism: Three-of-four translation consensus correction
evidence_used: Unanimous translation consensus increased correctness from 9,319 to 9,321 while lowering cross-entropy to 0.198013900, showing that translated views can identify beneficial prediction changes; three-view consensus tests whether slightly broader agreement captures additional corrections.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 74.58839158294722, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1979814826965332, "validation_score": 9319.417368721655}

RECENT RESULT
hypothesis: Accepting a changed prediction when at least three translated views support it and all four reject the base class will exceed 9,321 correct predictions.
change: Broaden unanimous correction only to three-of-four alternatives whose dissenting view also rejects the original prediction.
mechanism: Three-view consensus with unanimous base rejection
evidence_used: Unanimous consensus improved correctness from 9,319 to 9,321, while unrestricted three-of-four consensus fell to 9,319; requiring every view to reject the base class isolates a stronger subset of the added corrections.
result: training did not finish within the verification time limit



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
