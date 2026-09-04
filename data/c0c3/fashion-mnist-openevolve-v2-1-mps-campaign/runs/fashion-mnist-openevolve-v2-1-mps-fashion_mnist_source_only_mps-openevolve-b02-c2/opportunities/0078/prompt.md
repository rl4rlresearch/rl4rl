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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.09728554193862, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19226545028686523, "validation_score": 9320.419369696472}
prior_hypothesis: Restoring hard-maximum attention and weighting BatchNorm updates by actual batch size will exceed 9,322 correct predictions by retaining the 9,320-correct fast architecture while preventing the two 80-example tail batches from disproportionately shifting evaluation statistics.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 77.90509041585028, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19264950561523436, "validation_score": 9320.419234651627}
prior_hypothesis: Plain paired-view cross-entropy with mean-logit flip ensembling will exceed 9,322 correct predictions while retaining the reliably completed hard-maximum architecture’s runtime.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging four regional maxima will exceed 9,322 correct predictions by retaining top-four attention’s multi-peak robustness at near hard-maximum cost, while removing redundant preparation-time flipping provides runtime headroom.
change: Replace each channel’s single global maximum with the mean of a 2×2 adaptive-max map, fuse the linear attention calls, and pass preparation batches unchanged because paired-view loss already includes both orientations.
mechanism: Low-cost regional-peak channel saliency
evidence_used: Top-four saliency achieved the best verified result at 9,322 correct but repeatedly timed out, while global-maximum attention finished at 9,320; regional maxima provide a computationally cheaper four-peak descriptor.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the redundant preparation-time flip with deterministic one-pixel translations will exceed 9,322 correct predictions by improving positional invariance while retaining the completed hard-maximum architecture’s runtime.
change: Cycle each training batch through the nine one-pixel vertical/horizontal offsets using a single `torch.roll`; paired-view loss continues to provide both horizontal orientations.
mechanism: Low-overhead cyclic one-pixel translation augmentation
evidence_used: Hard-maximum attention completed with 9,320 correct, while independent replicated-border translations timed out; a shared one-pixel roll removes the redundant random-flip operations and tests the same positional-invariance hypothesis with substantially less preparation overhead.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 81.99823325010948, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21028710441589354, "validation_score": 9245.413125115665}

RECENT RESULT
hypothesis: Reweighting channel evidence toward the spatial mean while restoring plain paired-view cross-entropy will exceed 9,322 correct predictions without top-k’s runtime cost.
change: Replace equal average-plus-maximum attention with a magnitude-preserving 1.25:0.75 mean/maximum blend evaluated by one shared attention call, and remove the harmful ensemble-aware loss.
mechanism: Mean-tempered maximum channel saliency
evidence_used: Top-four saliency reached 9,322 correct versus 9,320 for hard maximum, indicating that tempering peak outliers helps, but top-k and RMS-tempered variants timed out; plain paired-view cross-entropy scored 9,320 while the ensemble-aware objective fell to 9,307.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging orientation logits at validation will exceed 9,322 correct predictions by requiring cross-orientation class agreement consistent with the paired-view cross-entropy objective.
change: Preserve the best top-four attention training design and replace arithmetic probability averaging with a faster geometric-probability ensemble implemented as mean logits.
mechanism: Loss-aligned geometric flip ensemble
evidence_used: The top-four model achieved 9,322 correct with independent paired-view cross-entropy, while adding an arithmetic-ensemble-aware loss reduced performance to 9,307; this motivates aligning inference with the successful individual-view log-loss rather than the harmful arithmetic objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging each channel’s two strongest activations will exceed 9,322 correct predictions by tempering single-pixel outliers less aggressively than top-four attention, while removing the redundant preparation flip offsets its runtime cost.
change: Replace hard global-maximum channel evidence with top-two averaging and make batch preparation a no-op because paired-view training already presents both orientations.
mechanism: Top-two channel saliency with redundant augmentation removal
evidence_used: Top-four saliency improved hard maximum from 9,320 to 9,322 correct, showing that multi-peak evidence helps; hard maximum finished faster at 75.3 seconds than top-four’s 78.8 seconds, motivating a smaller top-k and removal of redundant flip work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 0.99-per-step EMA sampled every eight steps will exceed 9,322 correct predictions by suppressing late-update noise while avoiding the timeout of the prior every-step EMA.
change: Restore ordinary BatchNorm momentum and maintain a mathematically time-adjusted parameter EMA every eight steps during the final half of training, installing it after the last update.
mechanism: Sparse final-half parameter EMA
evidence_used: Hard-maximum attention with ordinary BatchNorm finished in 75.3 seconds at 9,320 correct, while sample-weighted BatchNorm provided no accuracy gain; the prior every-step EMA timed out, motivating the same stabilization at one-eighth the update frequency.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring plain paired-view cross-entropy and raising the cosine schedule floor from 1% to 5% will exceed 9,322 correct predictions by preserving late-stage refinement without adding runtime-heavy operations.
change: Remove the harmful arithmetic-ensemble loss and retain more learning rate near the end of the fixed 100,000-example exposure.
mechanism: Higher terminal learning-rate floor with plain paired-view supervision
evidence_used: Plain paired-view cross-entropy with hard-maximum attention reached 9,320 correct, while the ensemble-aware objective reduced that to 9,307; the hard-maximum architecture also finished faster than the 9,322-correct top-four design, leaving schedule refinement as a low-cost orthogonal test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 73.99872791697271, "validation_accuracy": 0.9306, "validation_correct": 9306, "validation_cross_entropy": 0.19502095413208007, "validation_score": 9306.418402705218}

RECENT RESULT
hypothesis: Replacing the 1% learning-rate floor with pure cosine decay will exceed 9,322 correct predictions by reducing late-stage parameter noise without adding runtime.
change: Preserve the best verified top-four attention design and plain paired-view loss, changing only the cosine schedule to anneal the learning rate to zero.
mechanism: Zero-floor cosine annealing
evidence_used: Raising the cosine floor from 1% to 5% reduced the hard-maximum model from 9,320 to 9,306 correct, directly indicating that retaining more late-stage learning rate is harmful; lowering the floor is the clean, low-cost inverse test on the 9,322-correct design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing the proven 9,320-correct hard-maximum model to zero learning rate will exceed 9,322 correct predictions by reducing late-update noise while retaining its reliable runtime.
change: Replace the 1% terminal learning-rate floor with pure cosine decay, leaving the architecture, paired-view cross-entropy, and all other optimizer settings unchanged.
mechanism: Zero-floor cosine annealing on fast hard-maximum attention
evidence_used: Raising the floor from 1% to 5% reduced accuracy from 9,320 to 9,306 correct, indicating that additional late-stage learning is harmful; applying the inverse change to the faster hard-maximum design isolates the schedule effect without the timeout-prone top-four computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 83.43685666681267, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.19421031494140625, "validation_score": 9300.418686720206}

RECENT RESULT
hypothesis: Averaging each channel’s three strongest activations will exceed 9,322 correct predictions by interpolating between the 9,320-correct hard maximum and 9,322-correct top-four descriptor, while removing redundant work keeps verification within the time limit.
change: Replace hard-max evidence with top-three saliency, fuse the linear attention calls, remove redundant preparation flips, and restore ordinary BatchNorm momentum.
mechanism: Fused top-three channel saliency
evidence_used: Top-four saliency improved hard maximum by two correct predictions; sample-weighted BatchNorm did not improve accuracy, and paired-view training already presents both horizontal orientations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Plain paired-view cross-entropy with mean-logit flip ensembling will exceed 9,322 correct predictions while retaining the reliably completed hard-maximum architecture’s runtime.
change: Remove the harmful ensemble-aware training loss and replace arithmetic probability averaging at validation with geometric probability averaging implemented as mean logits.
mechanism: Loss-aligned geometric flip ensembling
evidence_used: Plain paired-view cross-entropy reached 9,320 correct versus 9,307 for the ensemble-aware objective; the prior geometric-ensemble test used the timeout-prone top-four model, so testing it on the 75.3-second hard-maximum design isolates the inference rule with runtime headroom.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 77.90509041585028, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19264950561523436, "validation_score": 9320.419234651627}

RECENT RESULT
hypothesis: Softening the successful arithmetic flip ensemble by temperature 1.05 will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.192551.
change: Divide validation-only ensemble logits by 1.05; training and predicted classes remain unchanged.
mechanism: Mild temperature calibration of flip-mixture logits
evidence_used: Arithmetic probability ensembling and geometric ensembling both produced 9,320 correct with hard-maximum attention, but arithmetic ensembling achieved lower cross-entropy (0.192261 versus 0.192650), showing that validation-logit calibration can improve the tie-breaker without changing correct count.
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
