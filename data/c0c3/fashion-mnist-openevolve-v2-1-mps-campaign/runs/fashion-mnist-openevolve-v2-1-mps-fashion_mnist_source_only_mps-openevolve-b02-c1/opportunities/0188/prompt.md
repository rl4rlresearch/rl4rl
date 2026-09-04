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
hypothesis: Centered-view weights of 2.28125 will preserve all 9,311 correct predictions while lowering cross-entropy below 0.1922469223, and batching each shifted view with its flip will allow verification to finish within the time limit.
change: Raise both centered-view weights from 2.25 to 2.28125, renormalize by 12.5625, and evaluate shifted/flipped view pairs together while preserving the original logit summation order.
mechanism: Fine-grained center-emphasized logit TTA with paired-view evaluation
evidence_used: Raising centered weights from 2.0 to 2.25 preserved 9,311 correct and improved cross-entropy; every finer upward trial timed out, motivating the smallest pending increase plus a lower-overhead equivalent TTA implementation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding residual skips across the existing same-width convolution pairs will change learned decision boundaries and exceed 9,311 correct predictions without increasing parameters or materially increasing verification time.
change: Reuse the existing 32- and 64-channel convolution pairs as post-activation residual blocks and route training and TTA through the residual encoder.
mechanism: Parameter-free intra-stage residual feature refinement
evidence_used: Temperature and TTA-weight tuning plateaued at 9,311 correct, while the added-convolution residual design timed out; parameter-free skips test residual refinement without its parameter, classifier-width, or computational costs.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 68.88485495815985, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.191316349029541, "validation_score": 9295.419703801099}

RECENT RESULT
hypothesis: Increasing the final interpolation toward the averaged tail weights from 0.5 to 0.75 will change learned decision boundaries and exceed 9,311 correct predictions without increasing runtime or parameters.
change: Apply 75% rather than 50% of the existing late-training parameter average at the final optimizer step.
mechanism: Stronger tail-weight averaging
evidence_used: Evaluation-temperature and TTA-weight tuning plateaued at 9,311 correct, while architectural changes either reduced accuracy or timed out; strengthening the already-computed tail average is a low-cost way to improve learned generalization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling centered training views 2.25 times as often as each shifted view will exceed 9,311 correct predictions by aligning learned invariances with the verified center-emphasized TTA ensemble.
change: Change augmentation sampling from a 2:1 center-to-each-shift ratio to the TTA-matched 2.25:1 ratio, using probabilities 9/25 for centered views and 4/25 for each one-pixel shift.
mechanism: Train–test geometric-view distribution alignment
evidence_used: Raising evaluation center weights from 2.0 to 2.25 preserved 9,311 correct while lowering cross-entropy to 0.1922469223, indicating a modest center emphasis is beneficial; applying that same ratio during learning can change decision boundaries without adding parameters or runtime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 62.08736362494528, "validation_accuracy": 0.9293, "validation_correct": 9293, "validation_cross_entropy": 0.19035666923522948, "validation_score": 9293.420042171329}

RECENT RESULT
hypothesis: Feeding global average and maximum statistics from the 64-channel intermediate feature map directly to the classifier will exceed 9,311 correct predictions by preserving texture and part evidence otherwise discarded by the final convolution and pooling stage.
change: Split the encoder into stages, concatenate the deepest spatial representation with intermediate channel-wise mean and maximum descriptors, and resize the hidden layer to remain below the parameter ceiling.
mechanism: Mid-level statistical feature bypass
evidence_used: Parameter-free residual skips fell to 9,295 correct while still classifying solely from the deepest feature map, suggesting that modifying feature refinement alone is insufficient. This patch challenges the shared assumption that all useful evidence must survive the final 96-channel spatial compression by creating a direct multi-scale path to class prediction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training with Beta(0.2, 0.2) mixup will learn smoother class boundaries and exceed 9,311 correct predictions without materially increasing verification time.
change: Mix each augmented image with a randomly paired batch image and optimize the correspondingly weighted pair of cross-entropy losses.
mechanism: Per-example mild mixup regularization
evidence_used: Evaluation-only TTA tuning plateaued at 9,311 correct, while residual architecture changes reduced accuracy and heavier mechanisms timed out; mixup changes learned boundaries using the existing single forward pass and no additional parameters.
result: the implementation could not be verified

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
