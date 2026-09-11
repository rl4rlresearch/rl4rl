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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.17715312493965, "validation_accuracy": 0.926, "validation_correct": 9260, "validation_cross_entropy": 0.21201698150634765, "validation_score": 9260.412535474032}
prior_hypothesis: An evaluation-logit multiplier of 1.4164 will preserve all 9,260 correct predictions while lowering validation cross-entropy below 0.2120170723.

## Recent verification evidence

RECENT RESULT
hypothesis: Holding the peak learning rate for the first 20% of training will increase validation_correct above 9,260 by allowing more optimization progress before decay under the fixed two-pass budget.
change: Replace immediate cosine decay with a 20%-hold followed by cosine decay over the remaining 80% of optimizer steps.
mechanism: Delayed cosine learning-rate decay
evidence_used: The prior test of this compute-neutral schedule timed out without validation evidence, while label-smoothing, augmentation, dropout, residual, and head-capacity changes all reduced validation correctness.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing label smoothing from 0.07 to 0.03 while retaining a 0.05 temporal average will exceed 9,260 correct predictions by combining stronger early regularization with stronger late class separation.
change: Replace fixed 0.05 label smoothing with a compute-neutral cosine schedule from 0.07 at initialization to 0.03 at the final training step.
mechanism: Cosine label-smoothing curriculum
evidence_used: Fixed 0.03 and 0.07 smoothing reduced validation_correct to 9,242 and 9,244 respectively, while fixed 0.05 achieved 9,260; scheduling symmetrically around 0.05 preserves the verified average strength while testing whether regularization timing, rather than its overall amount, improves the short two-pass optimization.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 83.48077733302489, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2106638572692871, "validation_score": 9252.412996553088}

RECENT RESULT
hypothesis: A 5% warmup from 20% to full learning rate followed by cosine decay will exceed 9,260 correct predictions by stabilizing early AdamW updates without materially reducing the fixed-budget learning-rate integral.
change: Initialize AdamW at 5e-4 and replace immediate cosine decay with a 5%-warmup, 95%-cosine schedule; preserve the verified architecture, loss, EMA, BatchNorm-buffer mixture, and evaluation calibration.
mechanism: Short linear learning-rate warmup with cosine decay
evidence_used: The 9,260-correct baseline uses immediate cosine decay, while label-smoothing, augmentation, dropout, residual, and capacity changes all reduced correctness. The only schedule experiment used a substantially more aggressive 20% peak hold and timed out, leaving a conservative compute-neutral warmup unresolved.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 75.30991720804013, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.20919652214050294, "validation_score": 9250.413497715917}

RECENT RESULT
hypothesis: Increasing the peak learning rate by 4% to 2.6e-3 will exceed 9,260 correct predictions by improving optimization progress without the aggressive schedule distortion of a peak hold.
change: Raise both AdamW’s initial learning rate and the cosine schedule’s base learning rate from 2.5e-3 to 2.6e-3, preserving all other settings.
mechanism: Modest cosine learning-rate rescaling
evidence_used: The 5% warmup reduced early learning rates and lowered validation_correct from 9,260 to 9,250 despite improving cross-entropy, suggesting that slightly more optimization progress may favor the primary correctness objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the peak and scheduled learning rate by 2% to 2.55e-3 will raise validation_correct above 9,260 by providing slightly more optimization progress without materially changing the verified training trajectory.
change: Raise AdamW’s initial learning rate and the cosine schedule’s base learning rate from 2.5e-3 to 2.55e-3, preserving all other settings.
mechanism: Conservative cosine learning-rate rescaling
evidence_used: A 5% warmup reduced early optimization and lowered validation_correct from 9,260 to 9,250; the larger 2.6e-3 test produced no validation evidence because verification timed out, motivating a more conservative midpoint test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 76.77867033309303, "validation_accuracy": 0.9248, "validation_correct": 9248, "validation_cross_entropy": 0.2126882438659668, "validation_score": 9248.412307122238}

RECENT RESULT
hypothesis: Retaining 10% of the final optimizer iterate when installing the EMA weights will exceed 9,260 correct predictions by preserving late decision-boundary refinement while retaining most EMA variance reduction.
change: Replace the pure EMA parameter copy with a 90% EMA, 10% terminal-weight interpolation; preserve architecture, optimizer, loss, schedule, BatchNorm mixture, and calibration.
mechanism: Terminal-weight/EMA interpolation
evidence_used: The immediate-cosine baseline remains strongest, while warmup improved cross-entropy but reduced correctness to 9,250 and both learning-rate increases also reduced correctness or timed out; this motivates a conservative change to late-weight averaging without perturbing the verified training trajectory.
result: training did not finish within the verification time limit

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
