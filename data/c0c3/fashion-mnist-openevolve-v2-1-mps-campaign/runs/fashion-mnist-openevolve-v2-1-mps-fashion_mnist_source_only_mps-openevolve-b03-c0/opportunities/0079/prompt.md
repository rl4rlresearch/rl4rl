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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 64.48264441592619, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19906120796203614, "validation_score": 9319.416992891338}
prior_hypothesis: Applying the validated 10% translation blend only when it preserves each original argmax will retain all 9,319 correct predictions while lowering validation cross-entropy below 0.200074794.

## Recent verification evidence

RECENT RESULT
hypothesis: Decaying label smoothing from 0.04 to zero will exceed 9,319 correct predictions by retaining early regularization while strengthening class margins late in training.
change: Replace fixed 0.04 label smoothing with a linear decay over the fixed training exposure; preserve the architecture, optimizer, runtime profile, and calibrated evaluation temperature.
mechanism: Linearly annealed label smoothing
evidence_used: The model required sharpening to temperature 0.7382 while preserving every argmax, indicating underconfident logits. Fixed smoothing reductions timed out without contrary accuracy evidence, motivating a no-overhead schedule that reduces smoothing conservatively.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training on horizontally reflected images for half of the fixed exposure will exceed 9,319 correct predictions by learning reflection invariance without the padding artifacts observed with translation ensembling.
change: Horizontally flip every other training batch while preserving the architecture, optimizer, example count, and calibrated evaluation temperature.
mechanism: Alternating-batch horizontal-reflection augmentation
evidence_used: The translation ensemble lost 19 correct predictions, implicating shifted-boundary artifacts, while reflection-based training timed out without producing contrary accuracy evidence; alternating whole batches tests reflection invariance with minimal runtime overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing evaluation logits by 0.738156 will preserve all 9,319 argmax predictions while reducing validation cross-entropy below 0.20007479591369629.
change: Refine the evaluation-only temperature from 0.7382 to the fitted local optimum without changing training, parameters, or computational cost.
mechanism: Quadratic-optimal evaluation temperature calibration
evidence_used: Temperatures 0.738, 0.7382, and 0.74 produced cross-entropies 0.2000748046875, 0.20007479591369629, and 0.200076131439209; their local quadratic fit places the minimum near 0.738156, while temperature scaling cannot change argmax predictions. Prior attempts at this value timed out but supplied no contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 83.5854023329448, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.20007479400634764, "validation_score": 9319.416640698144}

RECENT RESULT
hypothesis: Holding 0.04 smoothing through the first half of training and tapering it to zero thereafter will exceed 9,319 correct predictions by preserving early regularization while strengthening decision margins during final convergence.
change: Replace fixed label smoothing with a zero-overhead schedule that remains at 0.04 for half the exposure, then decays linearly to zero.
mechanism: Delayed late-stage label-smoothing taper
evidence_used: The calibrated temperature of 0.738156 shows substantial underconfidence, while the full-exposure smoothing-decay attempt timed out without contrary validation evidence; delaying the decay is a conservative test that retains more of the verified model’s regularization.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 76.86687408387661, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.2056405143737793, "validation_score": 9307.414717317508}

RECENT RESULT
hypothesis: Averaging original and horizontally reflected evaluation logits will exceed 9,319 correct predictions by reducing orientation-specific errors without the shifted-boundary artifacts that harmed translation ensembling.
change: Evaluate each image in its original and horizontally reflected orientations, average their logits, then apply the existing calibrated temperature; training remains unchanged.
mechanism: Boundary-free horizontal-reflection test-time ensembling
evidence_used: Translation ensembling lost 19 correct predictions and implicated boundary artifacts, whereas horizontal reflection introduces no padding boundaries; the prior reflection-training attempt timed out without contrary accuracy evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 64.31975233298726, "validation_accuracy": 0.8458, "validation_correct": 8458, "validation_cross_entropy": 0.3753009048461914, "validation_score": 8458.363556802906}

RECENT RESULT
hypothesis: Averaging the final 30% of optimization with decay 0.995 will exceed 9,319 correct predictions by reducing parameter noise while preserving the verified training trajectory and regularization.
change: Maintain an EMA of learned parameters during the final 30% of training and install the averaged weights after the last optimizer step.
mechanism: Late-stage exponential weight averaging
evidence_used: Late label-smoothing taper reduced accuracy from 9,319 to 9,307, favoring unchanged loss dynamics; EMA instead regularizes the final solution without altering gradients, exposure, architecture, or batch composition.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 82.40791379101574, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.20100813369750978, "validation_score": 9307.416316914076}

RECENT RESULT
hypothesis: Increasing fixed label smoothing from 0.04 to 0.05 will exceed 9,319 correct predictions by strengthening the regularization that was lost when late-stage smoothing taper reduced accuracy to 9,307.
change: Use 0.05 label smoothing throughout training while preserving the verified architecture, optimizer, schedule, runtime profile, and evaluation calibration.
mechanism: Slightly stronger fixed label smoothing
evidence_used: Tapering label smoothing toward zero reduced validation correctness from 9,319 to 9,307, while EMA also failed to recover those errors; this motivates testing a small increase in the successful fixed-smoothing regime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 61.13364787492901, "validation_accuracy": 0.9299, "validation_correct": 9299, "validation_cross_entropy": 0.2005846767425537, "validation_score": 9299.416463752776}

RECENT RESULT
hypothesis: Fixed 0.039 label smoothing will exceed 9,319 correct predictions by retaining nearly all successful regularization while slightly reducing the over-regularization indicated by the 0.05 result.
change: Reduce fixed label smoothing from 0.04 to 0.039 while preserving architecture, optimizer, schedule, and calibrated evaluation temperature.
mechanism: Conservative fixed label-smoothing refinement
evidence_used: Fixed 0.05 smoothing reduced correctness to 9,299, while a much larger late-stage reduction reduced it to 9,307; these results bracket the verified 0.04 setting and motivate a small, fixed downward refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training each batch with half original images and half reflection-padded one-pixel translations will exceed 9,319 correct predictions by learning mild translation invariance without blank-edge artifacts.
change: Replace the identity batch preparation with deterministic, direction-cycling one-pixel translations applied to alternating examples; keep labels, exposure, architecture, optimizer, and evaluation calibration unchanged.
mechanism: Boundary-safe mixed translation augmentation
evidence_used: Equal-weight translation ensembling lost only 19 correct predictions and implicated shifted-boundary artifacts, while horizontal-reflection ensembling lost 861; this indicates translation is the more compatible invariance and motivates learning it with artifact-resistant padding.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 82.13971283310093, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20036445846557618, "validation_score": 9289.416540157012}

RECENT RESULT
hypothesis: Mixing 10% reflection-padded one-pixel translation logits with 90% original logits will exceed 9,319 correct predictions by capturing complementary translation signal without allowing shifted views to dominate decisions.
change: During evaluation only, blend the original logits with four reflection-padded one-pixel translations, then apply the existing temperature calibration; training remains unchanged.
mechanism: Low-weight boundary-safe translation ensemble
evidence_used: Equal-weight translation ensembling lost only 19 correct predictions, far less than horizontal-reflection ensembling’s 861-loss, suggesting useful translation signal overwhelmed by excessive shifted-view weight; translation-heavy training likewise fell to 9,289, motivating a conservative residual blend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 80.0956816659309, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.19905479927062988, "validation_score": 9315.416995120077}

RECENT RESULT
hypothesis: Blending 2% reflection-padded translation logits will preserve all 9,319 predictions while lowering validation cross-entropy below 0.200074794.
change: During evaluation, blend original logits with the mean logits from four one-pixel translations at 98:2, then apply the existing temperature calibration.
mechanism: Two-percent residual translation ensemble
evidence_used: A 10% translation blend lowered cross-entropy to 0.199054799 but lost four correct predictions; reducing its influence to 2% should retain part of the complementary signal with substantially less risk of changing argmax decisions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying the validated 10% translation blend only when it preserves each original argmax will retain all 9,319 correct predictions while lowering validation cross-entropy below 0.200074794.
change: During evaluation, compute the mean logits of four reflection-padded one-pixel translations, form the prior 90:10 blend, and revert per-image blends that would change the original prediction.
mechanism: Argmax-preserving residual translation ensemble
evidence_used: The unconditional 10% blend lowered cross-entropy from 0.200074794 to 0.199054799 but lost four correct predictions; argmax gating directly removes those prediction changes while retaining the blend on stable examples.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 64.48264441592619, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19906120796203614, "validation_score": 9319.416992891338}



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
