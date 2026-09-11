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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 72.18467708397657, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1981589729309082, "validation_score": 9319.417306894407}
prior_hypothesis: A 28% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198185161.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Increasing the gated translation contribution from 10% to 20% will retain all 9,319 original predictions while lowering validation cross-entropy below 0.199061208.
change: Blend original and translated logits at 80:20 during evaluation, retaining the existing per-image argmax-preservation gate and temperature calibration.
mechanism: Argmax-preserving 20% translation blend
evidence_used: The gated 10% blend preserved all 9,319 correct predictions and reduced cross-entropy from 0.200074794 to 0.199061208; its nearly identical cross-entropy to the unconditional 10% blend shows that reverting changed predictions sacrificed little of the translation benefit, motivating a stronger safely gated blend.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the gated translation contribution to 20% will retain all 9,319 original predictions while reducing validation cross-entropy below 0.199061208.
change: Blend original and four-view translated logits at 80:20, retaining the per-image argmax-preservation gate and calibrated temperature.
mechanism: Argmax-preserving 20% translation blend
evidence_used: The gated 10% blend preserved 9,319 correct predictions and improved cross-entropy from 0.200074794 to 0.199061208; the prior 20% attempt timed out without contrary performance evidence, while the unchanged gate guarantees that blended logits cannot alter predicted classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 58.86245729099028, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1984780334472656, "validation_score": 9319.417195798376}

RECENT RESULT
hypothesis: A 28.55% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198478033.
change: Increase the evaluation-only translation blend from 20% to 28.55%, retaining the argmax-preservation gate and calibrated temperature.
mechanism: Quadratic-optimal argmax-preserving translation blend
evidence_used: Cross-entropy improved from 0.200074794 at 0% translation to 0.199061208 at 10% and 0.198478033 at 20%; a quadratic fit to these three verified points places the local minimum near 28.55%, while the gate prevents prediction changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 28.55% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198478033.
change: Increase the evaluation-only translation blend from 20% to 28.55%, retaining argmax gating and temperature calibration.
mechanism: Quadratic-optimal argmax-preserving translation blend
evidence_used: Verified cross-entropy improved from 0.200074794 at 0% translation to 0.199061208 at 10% and 0.198478033 at 20%; the fitted local optimum is approximately 28.55%, and the previous attempt timed out without contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 30% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198478033.
change: Increase the evaluation-only translated-logit blend from 20% to 30%, retaining argmax gating and temperature calibration.
mechanism: Argmax-preserving 30% translation blend
evidence_used: Cross-entropy improved monotonically from 0.200074794 at 0% translation to 0.199061208 at 10% and 0.198478033 at 20%; the fitted optimum is near 28.55%, while two 28.55% attempts timed out without contrary performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 25% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198478033.
change: Increase the evaluation-only translated-logit blend from 20% to 25%, retaining argmax gating and temperature calibration.
mechanism: Argmax-preserving 25% translation blend
evidence_used: Gated translation improved cross-entropy monotonically from 0.200074794 at 0% to 0.199061208 at 10% and 0.198478033 at 20% without changing correctness; 25% conservatively advances toward the fitted optimum near 28.55%, whose attempts timed out without contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 79.08743112511002, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.19829813995361328, "validation_score": 9319.417258429541}

RECENT RESULT
hypothesis: A 28% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198298140.
change: Increase the evaluation-only translated-logit blend from 25% to 28%, retaining argmax gating and temperature calibration.
mechanism: Near-optimal argmax-preserving translation blend
evidence_used: Increasing the gated blend from 20% to 25% preserved 9,319 correct predictions and reduced cross-entropy from 0.198478033 to 0.198298140; 28% advances toward the evidence-derived optimum near 28.55%, while the gate prevents prediction changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 27% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198298140.
change: Increase the evaluation-only translated-logit blend from 25% to 27%, retaining argmax gating and temperature calibration.
mechanism: Argmax-preserving 27% translation blend
evidence_used: Increasing the gated blend from 20% to 25% preserved 9,319 correct predictions and reduced cross-entropy from 0.198478033 to 0.198298140; 27% is a conservative step toward the fitted optimum near 28.55%, while the prior 28% timeout supplied no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 70.15812437492423, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1981851608276367, "validation_score": 9319.417297773622}

RECENT RESULT
hypothesis: A 28% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198185161.
change: Increase the evaluation-only translated-logit blend from 27% to 28%, retaining argmax gating and temperature calibration.
mechanism: Near-optimal argmax-preserving translation blend
evidence_used: The 27% blend preserved 9,319 correct predictions and improved cross-entropy from 0.198298140 at 25% to 0.198185161; 28% moves toward the evidence-derived optimum near 28.55%, and its prior timeout provided no contrary performance evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 72.18467708397657, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1981589729309082, "validation_score": 9319.417306894407}



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
