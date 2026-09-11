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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 42.50887762499042, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20954851112365722, "validation_score": 9278.413377384539}
prior_hypothesis: AdamW beta2=0.97 will exceed 9,277 correct predictions by continuing the improvement observed from beta2=0.99 to 0.98 while remaining stable over the fixed 522-step run.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 30.580157082993537, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.2100638172149658, "validation_score": 9277.413201347637}
prior_hypothesis: AdamW beta2=0.98 will exceed 9,268 correct predictions by extending the improvement from beta2=0.99 while remaining stable over the fixed 522-step run.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 34.46409162483178, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20936809196472167, "validation_score": 9283.4134390541}
prior_hypothesis: AdamW beta2=0.96 will exceed 9,278 correct predictions by continuing the verified improvement from beta2=0.99 through 0.98 to 0.97 while remaining stable over 522 optimizer steps.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.22171137481928, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20909543914794923, "validation_score": 9280.41353228522}
prior_hypothesis: AdamW beta2=0.95 will exceed 9,283 correct predictions by continuing the verified monotonic improvement from beta2=0.99 through 0.96 while remaining stable over 522 optimizer steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Redistributing the successful vertical radius-1 weight from the downward-content view to the upward-content view will exceed 9,268 correct predictions by exploiting directional alignment while preserving the accuracy-safe total vertical weight.
change: Restore the best distance-stratified radius-2 TTA and moderate vertical-over-horizontal weighting, then assign 0.11/0.09 weights to the two opposite vertical shifts instead of 0.10/0.10.
mechanism: Sign-aware vertical translation ensembling
evidence_used: Moderate vertical bias retained 9,268 correct and improved cross-entropy, whereas stronger symmetric vertical bias lost five predictions; this motivates testing vertical directionality without increasing total vertical emphasis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 40.76835841592401, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.2124018669128418, "validation_score": 9264.412404511775}

RECENT RESULT
hypothesis: Replacing both information-discarding max pools with space-to-depth phase preservation and learned channel mixing will exceed 9,268 correct predictions by retaining subpixel garment structure for the convolutional classifier.
change: Replace the shared max-pooling backbone with PixelUnshuffle-based learned polyphase reductions while retaining the successful spatial classifier, and restore the best verified vertical-biased TTA weights.
mechanism: Learned polyphase downsampling
evidence_used: Head replacement and global channel gating regressed to 9,227 and 9,257 correct while retaining the same max-pooled representation; meanwhile, prediction quality responded to one-pixel TTA orientation, motivating a clean test of whether preserving sampling-phase information before learned mixing improves the representation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241022, "training_seconds": 43.9022715408355, "validation_accuracy": 0.9207, "validation_correct": 9207, "validation_cross_entropy": 0.2177106430053711, "validation_score": 9207.410606577903}

RECENT RESULT
hypothesis: Favoring the downward-content radius-1 view over the upward-content view will exceed 9,268 correct predictions because the previously tested opposite asymmetry materially changed predictions and reduced cross-entropy, demonstrating useful vertical sign sensitivity but choosing the wrong directional preference.
change: Preserve the accuracy-safe total vertical TTA weight of 0.20 while assigning 0.11 to the downward-content shift and 0.09 to the upward-content shift; retain all other architecture, training, and TTA settings.
mechanism: Opposite sign-aware vertical translation ensembling
evidence_used: Symmetric 0.10/0.10 vertical weighting achieved 9,268 correct, while favoring the upward-content view produced 9,264 with substantially lower 0.2124019 cross-entropy; testing the complementary direction is the cleanest unresolved sign-aware comparison.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 35.01191437500529, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21249540481567383, "validation_score": 9267.412372696848}

RECENT RESULT
hypothesis: Averaging the final 32 low-learning-rate parameter states while retaining the best vertical-biased TTA will exceed Reference Design 3’s validation_score by reducing boundary instability without changing model capacity.
change: Add a tail-averaging AdamW optimizer that installs the mean of the final 32 parameter states before validation, and restore the verified 0.10 vertical/0.08 horizontal TTA weights.
mechanism: Late-trajectory parameter averaging
evidence_used: Reference Design 3 preserved 9,268 correct with the best cross-entropy, while architecture, pooling, gating, and augmentation changes regressed; this motivates preserving its representation and improving the final solution through low-risk trajectory averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 45.657704167068005, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21340101928710936, "validation_score": 9257.412064924994}

RECENT RESULT
hypothesis: Extending the successful moderate vertical preference to radius-2 TTA views will exceed 9,268 correct predictions while preserving total ensemble weight.
change: Restore Reference Design 3’s accuracy-safe radius-1 weights and distance-stratified radius-2 weights, then mildly favor vertically dominant radius-2 translations over horizontally dominant ones.
mechanism: Multiscale vertical-axis translation ensembling
evidence_used: Vertical-biased radius-1 TTA achieved the best verified tied score at 9,268 correct and 0.2124414 cross-entropy, while horizontal bias worsened cross-entropy and stronger radius-1 bias lost accuracy; radius-2 orientation remains untested.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.77745370892808, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21242407264709473, "validation_score": 9265.412396958523}

RECENT RESULT
hypothesis: AdamW beta2=0.99 will exceed 9,268 correct predictions by adapting gradient-variance estimates faster during the fixed 522-step training run while retaining the best verified architecture and TTA.
change: Restore Reference Design 3’s vertical-biased TTA and use a faster-decaying AdamW second-moment estimate.
mechanism: Short-horizon second-moment adaptation
evidence_used: Reference Design 3 achieved the best tied score with 9,268 correct; architecture, augmentation, and late-trajectory averaging changes regressed, motivating a clean optimization-dynamics test on the strongest verified representation.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 41.99500587489456, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21068120079040528, "validation_score": 9268.412990636736}

RECENT RESULT
hypothesis: AdamW beta2=0.98 will exceed 9,268 correct predictions by extending the improvement from beta2=0.99 while remaining stable over the fixed 522-step run.
change: Reduce AdamW beta2 from 0.99 to 0.98 while preserving the best architecture, schedule, augmentation, and vertical-biased TTA.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Changing beta2 from AdamW’s default 0.999 to 0.99 retained 9,268 correct and reduced validation cross-entropy from 0.2124414 to 0.2106812, the strongest verified optimizer improvement.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 30.580157082993537, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.2100638172149658, "validation_score": 9277.413201347637}

RECENT RESULT
hypothesis: AdamW beta2=0.97 will exceed 9,277 correct predictions by continuing the improvement observed from beta2=0.99 to 0.98 while remaining stable over the fixed 522-step run.
change: Reduce AdamW beta2 from its current default of 0.999 to 0.97 while preserving the strongest architecture, augmentation, schedule, and vertical-biased TTA.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Beta2=0.99 reduced cross-entropy to 0.2106812 without losing accuracy, and beta2=0.98 further improved to 9,277 correct with 0.2100638 cross-entropy, supporting a conservative continuation of the same optimizer trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 42.50887762499042, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20954851112365722, "validation_score": 9278.413377384539}

RECENT RESULT
hypothesis: AdamW beta2=0.96 will exceed 9,278 correct predictions by continuing the verified improvement from beta2=0.99 through 0.98 to 0.97 while remaining stable over 522 optimizer steps.
change: Restore the strongest vertical-biased TTA configuration and reduce AdamW beta2 to 0.96, preserving the architecture, augmentation, learning-rate schedule, and parameter count.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Beta2 reductions produced monotonic gains: 0.99 achieved 9,268 correct, 0.98 achieved 9,277, and 0.97 achieved 9,278 with the lowest cross-entropy of 0.2095485.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 34.46409162483178, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20936809196472167, "validation_score": 9283.4134390541}

RECENT RESULT
hypothesis: AdamW beta2=0.95 will exceed 9,283 correct predictions by continuing the verified monotonic improvement from beta2=0.99 through 0.96 while remaining stable over 522 optimizer steps.
change: Restore the strongest vertical-biased, distance-stratified TTA and reduce AdamW beta2 to 0.95 while preserving all other training and model settings.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Beta2 reductions improved validation from 9,268 correct at 0.99 to 9,277 at 0.98, 9,278 at 0.97, and 9,283 at 0.96, with cross-entropy decreasing monotonically to 0.2093681.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.22171137481928, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20909543914794923, "validation_score": 9280.41353228522}

RECENT RESULT
hypothesis: AdamW beta2=0.9575 will retain or exceed 9,283 correct predictions while lowering cross-entropy below 0.2093681.
change: Reduce AdamW beta2 from 0.96 to 0.9575 while preserving every other model, training, and evaluation setting.
mechanism: Bracketed second-moment decay interpolation
evidence_used: Beta2=0.96 achieved the best accuracy at 9,283 correct, while 0.95 lowered cross-entropy to 0.2090954 but lost three predictions; an interior value tests whether the cross-entropy gain can be captured without crossing the accuracy optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 46.96981837507337, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20925290603637695, "validation_score": 9280.413478435737}

RECENT RESULT
hypothesis: Adding a low-rank two-head spatial-attention block will exceed 9,283 correct predictions by learning content-dependent relationships among distant garment parts before spatial aggregation, while the residual scale preserves the strong convolutional baseline during short training.
change: Preserve the max-pooled convolutional representation, spatial classifier, augmentation, and TTA; insert compact relative-position self-attention at the 7×7 feature stage and restore the best verified AdamW beta2 of 0.96.
mechanism: Relative-position non-local part attention
evidence_used: Beta2=0.96 reached 9,283 correct, while further optimizer interpolation, pooling replacement, global channel gating, and head changes failed to improve it. This challenges the shared assumption that fixed local/dilated convolution alone adequately models long-range part relationships without discarding the verified backbone or classifier.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249489, "training_seconds": 40.61832020804286, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.20968514709472658, "validation_score": 9265.413330692867}



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
