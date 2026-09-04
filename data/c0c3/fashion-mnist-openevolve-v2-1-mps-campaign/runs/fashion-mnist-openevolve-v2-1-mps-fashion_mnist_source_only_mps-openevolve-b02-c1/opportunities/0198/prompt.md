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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 65.44625779101625, "validation_accuracy": 0.9324, "validation_correct": 9324, "validation_cross_entropy": 0.2011635368347168, "validation_score": 9324.416263052171}
prior_hypothesis: Reducing AdamW’s second-moment decay from 0.999 to 0.99 will exceed 9,318 correct predictions by adapting its preconditioner more quickly during the 2,084-step warmup-and-cosine schedule.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing label smoothing from 0.03 to 0.026 will exceed 9,318 correct predictions by retaining beneficial smoothing while avoiding the over-regularization observed at 0.05.
change: Set training-only cross-entropy label smoothing to 0.026; leave architecture, optimization, augmentation, and inference unchanged.
mechanism: Fine-grained target-distribution smoothing
evidence_used: Smoothing of 0.03 improved correctness from 9,311 to 9,318, while 0.05 fell to 9,312, placing the most promising region slightly below the successful 0.03 setting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing label smoothing from 0.03 to 0.028 will exceed 9,318 correct predictions by retaining the demonstrated smoothing benefit while modestly reducing over-regularization.
change: Set training-only cross-entropy label smoothing to 0.028, leaving architecture, optimization, augmentation, and inference unchanged.
mechanism: Fine-grained target-distribution smoothing
evidence_used: Label smoothing of 0.03 improved correctness from 9,311 to 9,318, while increasing it to 0.05 reduced correctness to 9,312; the 0.026 trial produced no contradictory metrics because it timed out, motivating a smaller downward refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Label smoothing of 0.025 will exceed 9,318 correct predictions by retaining the beneficial regularization of 0.03 while reducing the over-regularization observed at 0.05.
change: Reduce training-only cross-entropy label smoothing from 0.03 to 0.025; leave architecture, optimization, augmentation, and inference unchanged.
mechanism: Fine-grained target-distribution smoothing
evidence_used: Smoothing of 0.03 improved correctness from 9,311 to 9,318, while 0.05 reduced it to 9,312. The 0.026 and 0.028 trials produced no contradictory metrics because they timed out, leaving the predicted optimum near 0.025 unmeasured.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the asymmetric final 7×7-to-3×3 max pool with adaptive max pooling will exceed 9,318 correct predictions by retaining bottom/right-border features and improving translation consistency without added parameters or meaningful runtime cost.
change: Change only the final pooling layer from fixed 2×2 pooling, which ignores the last feature-map row and column, to 3×3 adaptive max pooling that covers the entire 7×7 map.
mechanism: Border-complete adaptive spatial pooling
evidence_used: The successful 0.03-label-smoothing design reached 9,318 correct with translation augmentation and TTA, while residual refinement reduced correctness; this motivates preserving that learned pipeline and addressing the final encoder’s spatial asymmetry directly.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 58.19390412513167, "validation_accuracy": 0.9314, "validation_correct": 9314, "validation_cross_entropy": 0.2023862190246582, "validation_score": 9314.415839762705}

RECENT RESULT
hypothesis: Input-conditioned channel recalibration at all three feature scales will exceed 9,318 correct predictions by learning multiplicative feature interactions that fixed feed-forward and residual representations cannot express.
change: Insert lightweight squeeze-excitation gates before every downsampling stage and narrow the classifier from 147 to 145 hidden units, keeping the model under the parameter ceiling at 249,843 parameters.
mechanism: Identity-initialized multi-scale channel attention
evidence_used: Residual refinement reached only 9,295 correct and border-complete adaptive max pooling reached 9,314, suggesting that additive feature refinement and pooling geometry are not the missing mechanism; identity-initialized channel attention instead lets global image context dynamically select features while preserving the successful baseline at initialization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging per-view class probabilities instead of unbounded logits will exceed 9,318 correct predictions by limiting the influence of confidently wrong shifted or flipped views.
change: Convert each TTA view’s logits to probabilities before applying the existing center weights, then return temperature-scaled log-probabilities as valid ten-class logits.
mechanism: Arithmetic posterior TTA aggregation
evidence_used: Label smoothing improved correctness from 9,311 to 9,318, indicating that reducing overconfident decisions is beneficial; probability-space aggregation applies the same principle specifically to disagreements within the existing multi-view ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 78.84117470914498, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.20651403427124024, "validation_score": 9315.414417060885}

RECENT RESULT
hypothesis: Applying 6×6 cutout to 25% of training images will exceed 9,318 correct predictions by improving partial-occlusion robustness while remaining milder than the over-regularizing 0.05 label smoothing.
change: Add training-only random 6×6 zero-filled erasure after the existing translation and horizontal-flip augmentation.
mechanism: Mild stochastic cutout regularization
evidence_used: Label smoothing of 0.03 improved correctness from 9,311 to 9,318, while stronger 0.05 smoothing fell to 9,312; this motivates a mild, complementary input-space regularizer rather than stronger target smoothing or altered inference aggregation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling smoothing up to 0.04 by detached true-class confidence will exceed 9,318 correct predictions by regularizing confident examples while preserving stronger corrective gradients on borderline errors.
change: Replace fixed 0.03 label smoothing with per-example smoothing equal to 0.04 times the model’s detached true-class probability.
mechanism: Confidence-gated label smoothing
evidence_used: Fixed 0.03 smoothing improved correctness from 9,311 to 9,318, while fixed 0.05 smoothing fell to 9,312, motivating selective rather than uniformly stronger smoothing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Label smoothing of 0.026 will exceed 9,318 correct predictions by retaining the gain at 0.03 while avoiding the over-regularization observed at 0.05.
change: Reduce training-only cross-entropy label smoothing from 0.03 to 0.026; leave architecture, optimization, augmentation, and inference unchanged.
mechanism: Quadratically tuned mild label smoothing
evidence_used: Smoothing improved correctness from 9,311 at the unsmoothed baseline to 9,318 at 0.03, then fell to 9,312 at 0.05; these measured points place the estimated optimum near 0.026, whose prior verification timed out without contradictory accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing AdamW’s second-moment decay from 0.999 to 0.99 will exceed 9,318 correct predictions by adapting its preconditioner more quickly during the 2,084-step warmup-and-cosine schedule.
change: Set AdamW betas to `(0.9, 0.99)` while preserving the model, loss, augmentation, learning-rate schedule, and weight averaging.
mechanism: Short-horizon Adam second-moment adaptation
evidence_used: Label-smoothing, pooling, and TTA variants failed to improve beyond 9,318 correct; with only 2,084 optimizer steps, the default 0.999 second-moment timescale spans much of training, motivating a distinct optimization-level change.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 65.44625779101625, "validation_accuracy": 0.9324, "validation_correct": 9324, "validation_cross_entropy": 0.2011635368347168, "validation_score": 9324.416263052171}

RECENT RESULT
hypothesis: Reducing AdamW’s second-moment decay from 0.99 to 0.98 will exceed 9,324 correct predictions by tracking gradient-scale changes more responsively during the 2,084-step warmup-and-cosine training horizon.
change: Change only AdamW’s second beta from 0.99 to 0.98, preserving the successful architecture, augmentation, loss, schedule, and tail weight averaging.
mechanism: Faster short-horizon Adam second-moment adaptation
evidence_used: Lowering beta2 from the default 0.999 to 0.99 improved correctness from 9,318 to 9,324 and cross-entropy from 0.20309 to 0.20116; continuing this successful optimization direction is the most informative next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 67.48404695908539, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.2030738510131836, "validation_score": 9311.41560208426}

RECENT RESULT
hypothesis: Increasing AdamW’s second beta from 0.99 to 0.992 will exceed 9,324 correct predictions by retaining faster adaptation while avoiding the instability observed at 0.98.
change: Change only AdamW’s second-moment decay from 0.99 to 0.992.
mechanism: Refined short-horizon Adam second-moment adaptation
evidence_used: Beta2=0.99 improved correctness from 9,318 to 9,324 and cross-entropy from 0.20309 to 0.20116, while beta2=0.98 fell to 9,311; this brackets the promising region near 0.99 and motivates a finer upward refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 79.97736287512816, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.2028072166442871, "validation_score": 9316.41569421357}



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
