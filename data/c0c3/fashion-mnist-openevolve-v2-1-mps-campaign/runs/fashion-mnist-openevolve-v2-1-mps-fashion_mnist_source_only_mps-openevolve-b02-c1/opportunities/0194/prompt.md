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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 77.31945699988864, "validation_accuracy": 0.9318, "validation_correct": 9318, "validation_cross_entropy": 0.2030936866760254, "validation_score": 9318.41559523214}
prior_hypothesis: Training with 0.03 label smoothing will exceed 9,311 correct predictions by reducing overconfidence on ambiguous examples without changing model capacity or runtime materially.

## Recent verification evidence

RECENT RESULT
hypothesis: A 0.15 training-only true-class logit margin will emphasize borderline errors and exceed 9,311 correct predictions without increasing model size or requiring extra forward passes.
change: Subtract 0.15 from each target logit before computing cross-entropy, leaving inference and TTA unchanged.
mechanism: Additive true-class margin training
evidence_used: Evaluation-only calibration plateaued at 9,311 correct, while residual refinement fell to 9,295 and train–test augmentation alignment fell to 9,293; this motivates a low-cost loss-level change that directly strengthens learned decision margins.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 76.30646858317778, "validation_accuracy": 0.9305, "validation_correct": 9305, "validation_cross_entropy": 0.1918720932006836, "validation_score": 9305.419508102297}

RECENT RESULT
hypothesis: Linearly removing head dropout during the final 20% of training will consolidate low-learning-rate features and tail-averaged weights, exceeding 9,311 correct predictions without added computation or parameters.
change: Keep dropout at 0.15 for the first 80% of training, then anneal it to zero over the same interval used for parameter averaging.
mechanism: Tail-phase dropout annealing
evidence_used: Evaluation-only tuning plateaued at 9,311 correct, while margin training and augmentation-distribution changes reduced correctness; this motivates a low-cost optimization change that preserves the successful model and early regularization while improving final convergence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training with 0.03 label smoothing will exceed 9,311 correct predictions by reducing overconfidence on ambiguous examples without changing model capacity or runtime materially.
change: Apply label smoothing only to the existing training cross-entropy; inference and the verified center-weighted logit ensemble remain unchanged.
mechanism: Mild label smoothing
evidence_used: The 0.15 true-class margin reduced correctness from 9,311 to 9,305, suggesting that further sharpening target-class pressure is harmful; mild smoothing tests the complementary loss-level direction at negligible computational cost.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 77.31945699988864, "validation_accuracy": 0.9318, "validation_correct": 9318, "validation_cross_entropy": 0.2030936866760254, "validation_score": 9318.41559523214}

RECENT RESULT
hypothesis: Increasing label smoothing from 0.03 to 0.05 will exceed 9,318 correct predictions by further improving ambiguous-class decision boundaries, despite potentially increasing validation cross-entropy.
change: Raise training-only cross-entropy label smoothing to 0.05 while leaving architecture, optimization, augmentation, and inference unchanged.
mechanism: Stronger target-distribution smoothing
evidence_used: Label smoothing of 0.03 improved correctness from 9,311 to 9,318, whereas a 0.15 true-class margin reduced it to 9,305; this supports testing a modestly stronger move in the successful smoothing direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 64.14228412508965, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.21601886672973633, "validation_score": 9312.411177830936}

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
