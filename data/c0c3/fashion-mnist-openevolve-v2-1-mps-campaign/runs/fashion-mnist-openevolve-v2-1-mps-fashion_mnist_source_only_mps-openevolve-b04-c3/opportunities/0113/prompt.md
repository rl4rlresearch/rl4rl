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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 73.68122837506235, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}
prior_hypothesis: Scaling ensemble logits by 1.03592 will preserve all 9,243 predictions while reducing cross-entropy below 0.20883301086425782.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 67.20907104178332, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883301086425782, "validation_score": 9243.413622059876}
prior_hypothesis: Scaling inference logits by 1.0360 will preserve exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20883404388427734.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.73991066706367, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20904548225402833, "validation_score": 9243.413549372079}
prior_hypothesis: A 0.583172607421875-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090454845428467.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.7220510840416, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}
prior_hypothesis: Restoring the verified 1.03592 inference scale will preserve 9,243 correct predictions and reduce cross-entropy below the current 0.20883301391601564.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.035897 will preserve all 9,243 predictions while reducing validation cross-entropy below 0.20883300514221192.
change: Refine only the positive inference-time logit scale from 1.03592 to 1.035897.
mechanism: Three-point local temperature interpolation
evidence_used: The verified scales 1.03577, 1.03592, and 1.036 produced cross-entropies 0.20883301391601564, 0.20883300514221192, and 0.20883301086425782 with identical correct counts; local quadratic interpolation places the minimum near 1.035897.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 56.386961499927565, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300971984864, "validation_score": 9243.413622060269}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.035946 will preserve all 9,243 predictions while reducing validation cross-entropy below 0.20883300514221192.
change: Change only the positive inference-time logit scale from 1.0360 to 1.035946.
mechanism: Convex local inference-temperature refinement
evidence_used: Scales 1.035897, 1.03592, and 1.0360 yielded cross-entropies 0.20883300971984864, 0.20883300514221192, and 0.20883301086425782 with identical correct counts; quadratic interpolation of these closest measurements places the local minimum near 1.035946.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.035946 will preserve all 9,243 predictions and reduce validation cross-entropy below 0.20883300514221192.
change: Change only the positive inference-time logit scale from 1.03577 to 1.035946.
mechanism: Quadratic inference-temperature refinement
evidence_used: The closest verified scales—1.035897, 1.03592, and 1.0360—give a quadratic minimum near 1.035946; its prior verification timed out and therefore provides no contradictory performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using power 0.5831695556640625 and scaling ensemble logits by 1.03592 will preserve 9,243 correct predictions while reducing validation cross-entropy from 0.20904548225402833 to approximately 0.20883300514221192.
change: Replace the current unscaled inference ensemble with the best verified power-mean order and positive logit scale; training remains unchanged.
mechanism: Verified power-mean flip ensembling with calibrated inference temperature
evidence_used: Reference Design 3 achieved the highest verified score, 9243.413622061835, with exactly this configuration and the same architecture and training procedure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding independent uniform ±1-pixel translations alongside horizontal flips will increase validation_correct above 9,243 after 100,000 examples without changing the 245,040-parameter model.
change: Generate one random 1-pixel crop from a replicate-padded image for every training example, then apply the existing random horizontal flip.
mechanism: Per-example integer translation augmentation
evidence_used: Repeated temperature and power-mean refinements preserved exactly 9,243 predictions and have reached a cross-entropy plateau near 0.208833005, motivating a training-time invariance change capable of improving class rankings.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the flip-ensemble logits by 1.03592 will preserve all 9,243 correct predictions while reducing validation cross-entropy below 0.20883301086425782.
change: Change only the inference-time logit scale from 1.0360 to the best verified value, leaving training and class rankings unchanged.
mechanism: Verified inference-temperature calibration
evidence_used: Reference Design 3 achieved the highest verified score, 9243.413622061835, with scale 1.03592 and cross-entropy 0.20883300514221192 using the otherwise identical implementation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified 1.03592 inference scale will preserve 9,243 correct predictions and reduce cross-entropy below the current 0.20883301391601564.
change: Change only the flip-ensemble logit scale from 1.03577 to 1.03592.
mechanism: Verified inference-temperature calibration
evidence_used: Reference Design 3 achieved the highest verified score, 9243.413622061835, with this exact scale and otherwise identical code; the current 1.03577 scale scored slightly lower.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 82.7220510840416, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.20883300514221192, "validation_score": 9243.413622061835}

RECENT RESULT
hypothesis: Using power 0.5831695556640625 and scaling inference logits by 1.03592 will preserve 9,243 correct predictions while reducing validation cross-entropy to approximately 0.208833005.
change: Apply the best verified power-mean order and inference-only logit scale without changing training.
mechanism: Verified power-mean flip ensembling with calibrated inference temperature
evidence_used: Reference Designs 2 and 3 both achieved the highest verified score, 9243.413622061835, with this exact inference configuration and otherwise identical code.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Ensembling centered and horizontally shifted views with their mirrors will increase validation_correct above 9,243 by reducing sensitivity to small horizontal alignment changes.
change: Replace the two-view inference ensemble with a six-view ensemble over the original image, replicated-edge ±1-pixel horizontal translations, and all three mirrored counterparts; training remains unchanged.
mechanism: Symmetric one-pixel translation and flip probability ensemble
evidence_used: Repeated power and temperature refinements preserved exactly 9,243 predictions, while the attempted training-time translation augmentation timed out without performance evidence; inference-only translation ensembling tests the same invariance without adding training work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining nearly the full spatial bottleneck while adding a learned covariance branch will increase validation_correct above 9,243 by capturing texture and channel co-occurrence distinctions that the current flattened first-order head cannot represent.
change: Reduce the existing hidden width from 38 to 37 and use the recovered parameter budget for a 24-channel centered covariance classifier, producing a 249,597-parameter model with minimal additional computation.
mechanism: Hybrid spatial-shape and second-order channel-covariance classifier
evidence_used: Repeated temperature and ensemble refinements plateaued at exactly 9,243 correct, indicating that calibration is no longer the limiting assumption. Unlike the unverifiable fixed 4×2 pooling experiment, this preserves the established spatial prediction path almost intact while testing a genuinely different, orderless second-order representation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the flip-ensemble logits by 1.03592 will preserve all 9,243 correct predictions while reducing validation cross-entropy below 0.20883301086425782.
change: Change only the inference-time logit scale from 1.0360 to 1.03592.
mechanism: Verified inference-temperature calibration
evidence_used: Reference Designs 2 and 3 achieved the best verified score, 9243.413622061835, with this exact scale and otherwise identical training; the current 1.0360 scale has slightly higher cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Using power 0.5831695556640625 and scaling inference logits by 1.03592 will preserve 9,243 correct predictions while reducing cross-entropy to approximately 0.208833005.
change: Restore the highest-scoring verified power-mean order and positive inference-only logit scale; training remains unchanged.
mechanism: Verified power-mean flip ensembling with calibrated inference temperature
evidence_used: Reference Designs 2 and 3 both achieved the best verified score, 9243.413622061835, with this exact inference configuration; an earlier restoration of scale 1.03592 also improved the objective and became an available design.
result: the implementation could not be verified



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
