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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 79.89010045793839, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200913772583008, "validation_score": 9260.412538143844}
prior_hypothesis: Increasing only the forced terminal EMA interpolation from 3% to 6% will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120130508.

## Recent verification evidence

RECENT RESULT
hypothesis: Updating the EMA every optimizer step with the four-step-equivalent decay rate will exceed 9,260 correct predictions by averaging the same temporal horizon without aliasing three of every four late-training iterates.
change: Replace quarter-rate EMA updates at 0.03 interpolation with per-step updates at 0.007586, preserving the effective decay across each four-step interval.
mechanism: Equivalent-decay dense EMA sampling
evidence_used: The 9,260-correct baseline remains strongest, while terminal-weight interpolation produced no validation evidence and broader changes reduced correctness; refining the verified pure-EMA trajectory without changing its effective averaging horizon is the most conservative untested lever.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the sparse EMA interpolation rate from 0.03 to 0.04 will exceed 9,260 correct predictions by retaining more late decision-boundary refinement without altering the training trajectory or adding per-step overhead.
change: Keep quarter-rate EMA updates but shorten their averaging horizon by increasing the interpolation rate to 0.04.
mechanism: Shorter-horizon sparse EMA
evidence_used: The immediate-cosine baseline remains strongest, while warmup reduced correctness and terminal-weight interpolation produced no validation evidence; the per-step EMA experiment also timed out, motivating a runtime-neutral test of later-weight emphasis within the verified sparse EMA procedure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging original-view and horizontally reflected-view logits will exceed 9,260 correct predictions by reducing orientation-sensitive errors without perturbing the verified training trajectory.
change: During evaluation only, average logits from each image and its horizontal reflection before applying the verified 1.4164 calibration multiplier.
mechanism: Evaluation-time horizontal-reflection logit ensemble
evidence_used: Training-time augmentation reduced correctness, while optimization, loss, capacity, and EMA changes either regressed or timed out; an evaluation-only ensemble tests invariance without changing learned weights or training dynamics.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 66.20655866595916, "validation_accuracy": 0.8876, "validation_correct": 8876, "validation_cross_entropy": 0.3303643898010254, "validation_score": 8876.375836878853}

RECENT RESULT
hypothesis: Original-weighted averaging with four one-pixel translated views will exceed 9,260 correct predictions by reducing dense-head sensitivity to minor image alignment while preserving the original prediction as half of the ensemble.
change: During evaluation, combine original-view logits at 50% weight with replicate-padded one-pixel left, right, up, and down translations, then apply the verified calibration multiplier; training remains unchanged.
mechanism: Conservative one-pixel translation logit ensemble
evidence_used: Horizontal-reflection ensembling reduced correctness from 9,260 to 8,876, showing that transformation semantics are critical; one-pixel translations are class-preserving and test spatial robustness without perturbing the verified training trajectory.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 71.47630274994299, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.21078011703491212, "validation_score": 9258.412956896935}

RECENT RESULT
hypothesis: Weighting the verified four-shift ensemble at only 20% will preserve or exceed the baseline’s 9,260 correct predictions while retaining some of the cross-entropy improvement observed with the overly strong 50% ensemble.
change: During evaluation, blend 80% original-view logits with 20% mean logits from four replicate-padded one-pixel translations, then apply the existing calibration multiplier.
mechanism: Low-weight translation test-time ensemble
evidence_used: The 50%-weighted translation ensemble reached 9,258 correct—only two below baseline—while improving cross-entropy from 0.212017 to 0.210780, motivating a smaller interpolation that perturbs fewer decisions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 90% original logits with 10% mean logits from four one-pixel translations will preserve all 9,260 correct predictions while lowering validation cross-entropy below 0.212017.
change: During evaluation, compute original and four replicate-padded translated views in one batched forward pass, blend their logits at 90%/10%, then apply the existing calibration multiplier.
mechanism: Low-weight translation test-time ensemble
evidence_used: The 50%-weighted translation ensemble improved cross-entropy to 0.210780 while losing only two correct predictions; reducing its influence to 10% should retain more of that calibration benefit with substantially less risk of crossing decision boundaries. The planned 20% test timed out without validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the evaluation-logit multiplier from 1.4164 to 1.4166 will preserve all 9,260 predictions while lowering validation cross-entropy below 0.2120169815.
change: Raise only the positive evaluation-logit multiplier by 0.0002, leaving training and class decisions unchanged.
mechanism: Fine-grained evaluation temperature calibration
evidence_used: Moving the multiplier from the prior 1.4160 setting to 1.4164 lowered cross-entropy from 0.2120170723 to 0.2120169815 while retaining 9,260 correct, indicating that the local cross-entropy slope still favored slightly sharper logits.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 76.9815094168298, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201700172424318, "validation_score": 9260.41253546715}

RECENT RESULT
hypothesis: An evaluation-logit multiplier of 1.416408 will preserve all 9,260 predictions while lowering validation cross-entropy below 0.212016981506.
change: Refine only the evaluation-logit multiplier from 1.4164 to 1.416408, leaving training and class decisions unchanged.
mechanism: Quadratic temperature interpolation
evidence_used: Multipliers 1.4160, 1.4164, and 1.4166 produced cross-entropies 0.2120170723, 0.212016981506, and 0.212017001724 with 9,260 correct; quadratic interpolation places the local minimum near 1.416408.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.86813062499277, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201698303222657, "validation_score": 9260.412535473512}

RECENT RESULT
hypothesis: Incorporating the terminal optimizer iterate into the sparse EMA with a conservative 3% interpolation will exceed 9,260 correct predictions by capturing late decision-boundary refinement without materially reducing EMA smoothing.
change: Force one final EMA update when training completes, while preserving the architecture, optimizer, schedule, BatchNorm mixture, and evaluation calibration.
mechanism: Final-iterate EMA inclusion
evidence_used: The verified sparse EMA reaches 9,260 correct but its four-step update phase omits the final three iterates; a larger 10% terminal interpolation timed out without validation evidence, motivating this smaller runtime-neutral test.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 63.41994745796546, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201305084228517, "validation_score": 9260.41253681192}

RECENT RESULT
hypothesis: Increasing only the forced terminal EMA interpolation from 3% to 6% will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120130508.
change: Use a 6% EMA interpolation for the final optimizer iterate while retaining 3% for regular quarter-rate EMA updates.
mechanism: Final-iterate EMA dose refinement
evidence_used: The newly verified 3% final-iterate inclusion improved the objective at unchanged accuracy, while the earlier 10% experiment timed out without validation evidence; 6% is a runtime-neutral intermediate test of the observed benefit.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 79.89010045793839, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21200913772583008, "validation_score": 9260.412538143844}

RECENT RESULT
hypothesis: Increasing the forced terminal EMA interpolation from 6% to 9% will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120091377.
change: Use a 9% EMA interpolation for the final optimizer iterate while retaining 3% for regular quarter-rate EMA updates.
mechanism: Final-iterate EMA dose continuation
evidence_used: Raising the terminal interpolation from 3% to 6% preserved 9,260 correct predictions and reduced cross-entropy from 0.2120130508 to 0.2120091377, motivating another runtime-neutral step in the same direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 7.5% terminal EMA interpolation will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120091377.
change: Increase only the forced final-iterate EMA interpolation from 6% to 7.5%; retain 3% quarter-rate EMA updates and all other training behavior.
mechanism: Intermediate final-iterate EMA interpolation
evidence_used: Increasing terminal interpolation from 3% to 6% preserved 9,260 correct and reduced cross-entropy from 0.2120130508 to 0.2120091377. The 9% attempt timed out without contrary validation evidence, so 7.5% is a conservative intermediate continuation.
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
