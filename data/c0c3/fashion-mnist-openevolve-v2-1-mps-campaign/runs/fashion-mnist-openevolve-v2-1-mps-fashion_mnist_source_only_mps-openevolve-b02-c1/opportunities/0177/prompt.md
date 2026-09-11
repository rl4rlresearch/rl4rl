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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 73.81380458292551, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.1922469223022461, "validation_score": 9311.41937621364}
prior_hypothesis: Raising both centered-view weights from 2.0 to 2.25 will change fewer borderline decisions than the unverified 2.5 trial while correcting enough crop-sensitive errors to exceed 9,311 correct predictions.

## Recent verification evidence

RECENT RESULT
hypothesis: Temperature 0.9351 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922865562.
change: Raise only the evaluation-time ensemble temperature from 0.9350 to 0.9351.
mechanism: Quadratic-fit ensemble temperature calibration
evidence_used: Verified temperatures from 0.9330 through 0.9350 preserved exactly 9,311 correct predictions while showing diminishing cross-entropy improvements; interpolation of those measurements places the apparent minimum near 0.9351, while the 0.93525 and 0.9355 attempts timed out without contradictory metrics.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Temperature 0.9351 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922865562.
change: Raise only the evaluation-time ensemble temperature from 0.9350 to 0.9351.
mechanism: Quadratic-fit ensemble temperature calibration
evidence_used: Every verified increase from 0.9330 through 0.9350 reduced cross-entropy without changing predictions; interpolation places the minimum near 0.9351, whose prior verification timed out without contradictory metric evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 78.68543058307841, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.19228655624389648, "validation_score": 9311.419362272754}

RECENT RESULT
hypothesis: Reallocating 6,752 parameters to a residual 7×7 feature-refinement block while slightly narrowing the dense head will exceed 9,311 correct predictions without exceeding the parameter ceiling.
change: Add a lightweight pointwise–depthwise–pointwise residual block before the final pooling layer and reduce the classifier width from 147 to 139.
mechanism: Bottleneck residual spatial refinement
evidence_used: Evaluation-temperature tuning plateaued at exactly 9,311 correct, including identical results at 0.9350 and 0.9351, so improving the primary objective requires changing learned predictions; this targeted reallocation tests residual spatial refinement without the complexity of the unverified multiscale redesign.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging per-view class probabilities instead of logits will suppress isolated overconfident TTA errors and exceed 9,311 correct predictions while preserving temperature calibration.
change: Replace evaluation-time weighted logit averaging with the same 12-weight arithmetic probability ensemble, converted back to calibrated logits.
mechanism: Arithmetic probability TTA aggregation
evidence_used: Temperature tuning plateaued at exactly 9,311 correct, so progress requires changed argmax decisions; this isolates the ensemble aggregation rule without altering training cost, parameters, or learned weights.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 82.18782612495124, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.19352132568359376, "validation_score": 9310.418928417315}

RECENT RESULT
hypothesis: Increasing both centered-view weights from 2 to 3 while retaining all shifted views will exceed 9,311 correct predictions by reducing decisions driven by border-discarding translated crops.
change: Reweight the existing logit ensemble toward the two centered views and renormalize its calibrated denominator from 12 to 14.
mechanism: Center-emphasized geometric TTA ensemble
evidence_used: Arithmetic probability aggregation reduced accuracy to 9,310, supporting retention of geometric logit aggregation; temperature tuning cannot change the 9,311 argmax plateau, so view weighting is the smallest isolated change capable of improving the primary objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing each class’s single affine output with four learned subtype templates will exceed 9,311 correct predictions by modeling multimodal within-class appearances while remaining under the parameter ceiling.
change: Narrow the embedding from 147 to 142 units and replace the ten-output linear layer with 40 component logits, aggregating four components per class through log-mean-exp during training and TTA.
mechanism: Latent-subtype log-mean-exp classifier
evidence_used: Temperature calibration plateaued at exactly 9,311 correct and probability-level TTA reduced correctness, so progress requires different learned decision boundaries. The current design’s load-bearing assumption is one affine template per class; this patch tests smooth multi-region class scores without the costly residual, attention, bilinear, or multiscale mechanisms that failed to verify.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising both centered-view weights from 2.0 to 2.5 will correct more borderline translation-sensitive decisions than it harms, exceeding 9,311 correct predictions.
change: Increase the original and horizontally flipped centered-view weights to 2.5 and renormalize the calibrated ensemble by its new total weight of 13.
mechanism: Moderate center-emphasized geometric TTA
evidence_used: Probability averaging fell to 9,310 correct, favoring logit aggregation; the stronger center-weighting trial timed out without contradictory metrics, so testing its midpoint is the smallest informative change capable of breaking the 9,311 argmax plateau.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising both centered-view weights from 2.0 to 2.25 will change fewer borderline decisions than the unverified 2.5 trial while correcting enough crop-sensitive errors to exceed 9,311 correct predictions.
change: Increase the original and horizontally flipped centered-view weights to 2.25 and renormalize the calibrated ensemble by its total weight of 12.5.
mechanism: Conservative center-emphasized geometric TTA
evidence_used: Probability averaging reduced correctness to 9,310, supporting continued logit aggregation; temperature calibration cannot change the 9,311 argmax plateau, while stronger center weighting timed out without contradictory metric evidence, motivating a smaller isolated reweighting.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 249601, "training_seconds": 73.81380458292551, "validation_accuracy": 0.9311, "validation_correct": 9311, "validation_cross_entropy": 0.1922469223022461, "validation_score": 9311.41937621364}

RECENT RESULT
hypothesis: Increasing both centered-view weights from 2.25 to 2.375 will either exceed 9,311 correct predictions by crossing additional translation-sensitive decision boundaries or preserve 9,311 correct while reducing cross-entropy below 0.1922469223.
change: Raise the original and horizontally flipped centered-view weights to 2.375 and renormalize the logit ensemble by its total weight of 12.75.
mechanism: Intermediate center-emphasized geometric TTA
evidence_used: Moving centered-view weights from 2.0 to 2.25 preserved 9,311 correct while improving cross-entropy from 0.1922865562 to 0.1922469223; the 2.5 trial timed out without contradictory metrics, motivating the intermediate weight.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising both centered-view weights from 2.25 to 2.3125 will preserve or exceed 9,311 correct predictions while reducing validation cross-entropy below 0.1922469223.
change: Increase the original and horizontally flipped centered-view weights to 2.3125 and renormalize the ensemble by its total weight of 12.625.
mechanism: Fine-grained center-emphasized geometric TTA
evidence_used: Raising centered weights from 2.0 to 2.25 preserved 9,311 correct while improving cross-entropy from 0.1922865562 to 0.1922469223; the 2.375 verification timed out without contradictory metrics, motivating the midpoint.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Centered-view weights of 2.28125 will retain or exceed 9,311 correct predictions while lowering validation cross-entropy below 0.1922469223.
change: Raise both centered-view weights from 2.25 to 2.28125 and renormalize the ensemble by its total weight of 12.5625.
mechanism: Fine-grained center-emphasized logit TTA
evidence_used: Increasing centered weights from 2.0 to 2.25 preserved 9,311 correct while improving cross-entropy; the 2.3125 attempt timed out without contradictory metrics, motivating its midpoint with the verified design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Centered-view weights of 2.28125 will preserve all 9,311 correct predictions while lowering cross-entropy below 0.1922469223, and batching each shifted view with its flip will allow verification to finish within the time limit.
change: Raise both centered-view weights from 2.25 to 2.28125, renormalize by 12.5625, and evaluate shifted/flipped view pairs together while preserving the original logit summation order.
mechanism: Fine-grained center-emphasized logit TTA with paired-view evaluation
evidence_used: Raising centered weights from 2.0 to 2.25 preserved 9,311 correct and improved cross-entropy; every finer upward trial timed out, motivating the smallest pending increase plus a lower-overhead equivalent TTA implementation.
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
