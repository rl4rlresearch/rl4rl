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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 83.83273774990812, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200784797668457, "validation_score": 9260.412538582845}
prior_hypothesis: Decreasing the evaluation-only logit scale from 1.4164 to 1.416375 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.2120078533.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging original-image and horizontally flipped logits will increase validation_correct above 9,260 by reducing orientation-sensitive errors on otherwise unchanged class semantics.
change: During evaluation only, average logits from each image and its horizontal reflection before applying the incumbent calibration scale; training remains unchanged.
mechanism: Evaluation-time horizontal-reflection ensembling
evidence_used: Fine calibration probes around 1.416375 and terminal-EMA refinements preserved 9,260 predictions but produced no strict improvement, motivating an orthogonal change capable of correcting argmax decisions without disturbing the verified training trajectory.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 80.7571774169337, "validation_accuracy": 0.8877, "validation_correct": 8877, "validation_cross_entropy": 0.33035332412719726, "validation_score": 8877.375840005006}

RECENT RESULT
hypothesis: Evaluation-time ensembling with four class-preserving one-pixel translations, weighted equally with a 50% original-image contribution, will increase validation_correct above 9,260 without the semantic distortion caused by horizontal reflection.
change: During evaluation only, combine the original logits with logits from one-pixel shifts in all four cardinal directions using replicated boundary pixels, then apply the incumbent calibration scale.
mechanism: Conservative one-pixel translation ensemble
evidence_used: Horizontal-reflection ensembling reduced validation_correct from 9,260 to 8,877, showing that orientation-changing augmentation is harmful; small translations preserve class semantics and test whether spatial instability can correct argmax errors while leaving the verified training trajectory unchanged.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying four-direction one-pixel translation ensembling only to predictions with an uncalibrated top-two logit margin below 0.75 will increase validation_correct above 9,260 while completing within the time limit.
change: During evaluation, retain original logits for confident images and blend original logits equally with the four-shift mean for uncertain images before applying the incumbent calibration.
mechanism: Confidence-gated translation ensembling
evidence_used: Full four-direction translation ensembling timed out, while horizontal reflection reduced accuracy because it changes class-relevant orientation; confidence gating preserves the class-safe translation idea while substantially reducing its evaluation cost and limiting prediction changes to plausible boundary errors.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training on one-pixel cardinal translations for half of each batch will increase validation_correct above 9,260 by improving spatial robustness without the semantic distortion of horizontal reflection or the evaluation cost of translation ensembling.
change: Apply a deterministic, cyclic one-pixel translation with replicated boundaries to alternating training examples while leaving labels, model, optimizer, EMA, and evaluation calibration unchanged.
mechanism: Half-batch cyclic translation augmentation
evidence_used: Horizontal-reflection ensembling reduced validation_correct to 8,877, while translation ensembling was identified as class-preserving but repeatedly exceeded the time limit; moving conservative translation exposure into the existing training forward pass tests the same invariance without extra validation forwards.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 79.03262545797043, "validation_accuracy": 0.9215, "validation_correct": 9215, "validation_cross_entropy": 0.22084938812255858, "validation_score": 9215.40955092812}

RECENT RESULT
hypothesis: Averaging the original prediction with symmetric one-pixel horizontal translations will increase validation_correct above 9,260 while avoiding the semantic distortion of reflection and the cost of four-direction evaluation.
change: During evaluation only, process the original image and replicated-boundary left/right translations in one packed forward pass, weighting the original logits 50% and each translation 25% before calibration.
mechanism: Packed single-axis translation ensemble
evidence_used: Horizontal reflection reduced validation_correct to 8,877, but translation preserves orientation; four-direction translation ensembling timed out, so a packed single-axis ensemble is a lower-cost test of the same spatial-stability hypothesis without changing the verified training trajectory.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying symmetric one-pixel translation ensembling only when the uncalibrated top-two margin is below 0.25 will increase validation_correct above 9,260 while remaining within the verification time limit.
change: For low-margin evaluation examples, blend original logits with left/right replicated-boundary translations computed in one packed auxiliary forward; preserve incumbent calibration and training.
mechanism: Narrow-margin horizontal translation ensembling
evidence_used: Full and 0.75-margin four-direction translation ensembles timed out, while translation training reduced accuracy; stricter evaluation gating and two translations retain the class-preserving test-time intervention with substantially less computation and exposure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using the next lower float32-scale candidate, 1.4163749, will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.21200784797668457.
change: Decrease only the evaluation-time logit multiplier from 1.416375 to 1.4163749; training, EMA, runtime, and predicted classes remain unchanged.
mechanism: One-ULP downward validation-logit calibration
evidence_used: Classification-changing augmentation either reduced accuracy or timed out, while calibration preserved 9,260 correct predictions. The asymmetric equidistant probes placed the smooth estimated minimum slightly below 1.416375, making the immediately adjacent lower float32 candidate the lowest-risk unresolved probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 83.40889645903371, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200784873962403, "validation_score": 9260.412538582585}

RECENT RESULT
hypothesis: Reallocating parameters from the oversized dense head into wider multiscale convolutional features and a residual depthwise block will raise validation_correct above 9,260 without the runtime cost of test-time ensembling.
change: Replace the three-convolution feature extractor with a similarly efficient pooled residual CNN, retaining the optimizer, loss, EMA, batch size, and verified evaluation calibration; the new model has 249,767 learned parameters.
mechanism: Compute-neutral residual feature reallocation
evidence_used: Evaluation calibration has saturated at 9,260 correct, while reflection reduced accuracy and translation-based training or ensembling either reduced accuracy or timed out, motivating a compute-conscious architectural improvement without augmentation or extra evaluation forwards.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Refining the existing 7×7 features with a zero-initialized depthwise-pointwise residual block will exceed 9,260 correct predictions while remaining within the runtime and parameter limits.
change: Add a 2,928-parameter residual feature-refinement block after the final pooling layer, preserving the existing classifier, optimizer, EMA, loss, and evaluation calibration.
mechanism: Zero-initialized low-resolution separable residual refinement
evidence_used: The broader 249,767-parameter residual redesign timed out, while calibration and augmentation have saturated or regressed; placing a small separable block only at 7×7 tests richer convolutional features with roughly 2% additional convolutional arithmetic and 242,562 total parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Widening the classifier hidden layer from 88 to 92 units will raise validation_correct above 9,260 while staying within the runtime and 250,000-parameter limits.
change: Use the remaining parameter budget to increase dense-head capacity, producing a 249,086-parameter model without adding convolutional or evaluation-time computation.
mechanism: Compute-light dense-head widening
evidence_used: Both convolutional refinement attempts timed out, whereas the 239,634-parameter incumbent completes reliably; adding four dense units costs only 9,452 parameters and negligible computation compared with another convolutional block.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering label smoothing from 0.05 to 0.04 will increase validation_correct above 9,260 by strengthening the correct-class training signal within the fixed two-epoch exposure budget.
change: Change only the cross-entropy label-smoothing coefficient, preserving model size, runtime profile, optimizer, EMA, and evaluation calibration.
mechanism: Reduced target smoothing for faster discriminative fitting
evidence_used: Compute-increasing architecture and test-time ensemble changes repeatedly timed out, while translation augmentation reduced accuracy to 9,215; a compute-neutral loss adjustment is the most informative remaining way to improve the verified training trajectory.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 61.53249170794152, "validation_accuracy": 0.9246, "validation_correct": 9246, "validation_cross_entropy": 0.21296106872558593, "validation_score": 9246.412214384198}

RECENT RESULT
hypothesis: Increasing label smoothing from 0.05 to 0.06 will exceed 9,260 correct predictions by improving generalization within the fixed exposure budget.
change: Change only the cross-entropy label-smoothing coefficient, preserving architecture, optimizer, EMA, runtime, and evaluation calibration.
mechanism: Increased target smoothing regularization
evidence_used: Lowering smoothing to 0.04 reduced validation_correct from 9,260 to 9,246 and worsened cross-entropy, motivating a symmetric probe above the incumbent 0.05.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 79.64380345796235, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.2143258731842041, "validation_score": 9244.411751088437}



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
