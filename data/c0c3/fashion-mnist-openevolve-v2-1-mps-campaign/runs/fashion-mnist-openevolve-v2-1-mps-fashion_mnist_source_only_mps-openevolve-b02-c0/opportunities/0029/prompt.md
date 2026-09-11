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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 65.0580692498479, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.23624278297424317, "validation_score": 9249.404451299442}
prior_hypothesis: Removing diagonal inference views will exceed 9,238 correct predictions by aligning evaluation with the successful cardinal-only crop distribution and avoiding weak out-of-distribution diagonal shifts.

## Recent verification evidence

RECENT RESULT
hypothesis: A per-axis 1:3:1 crop distribution will exceed 9,221 correct predictions by interpolating between the best 1:2:1 weighting and the slightly worse, more center-concentrated 1:4:1 weighting.
change: Increase the centered per-axis crop probability from 1/2 to 3/5 while preserving one-pixel translations and every other training and inference setting.
mechanism: Intermediate center-weighted translation augmentation
evidence_used: The 1:2:1 distribution achieved 9,221 correct, while increasing directly to 1:4:1 produced 9,220; testing the unmeasured intermediate 1:3:1 distribution is the most targeted refinement of the demonstrated center-weighting effect.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Vectorized 1:2:1-weighted inference will exceed 9,221 correct predictions by aligning test-time translation averaging with the best-performing crop distribution while avoiding the sequential implementation’s timeout.
change: Evaluate all translated and flipped views in one forward pass, weighting the center, cardinal, and diagonal translations 4:2:1.
mechanism: Vectorized center-weighted translation ensemble
evidence_used: Training with 1:2:1 crop weighting improved validation_correct from 9,203 to 9,221; the corresponding weighted inference attempt timed out, while vectorizing a larger 26-view ensemble completed successfully.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 70.20760804088786, "validation_accuracy": 0.9217, "validation_correct": 9217, "validation_cross_entropy": 0.23768644027709962, "validation_score": 9217.403979540964}

RECENT RESULT
hypothesis: Weighting center, cardinal, and diagonal inference views by 2:√2:1 will exceed 9,221 correct predictions by balancing the higher accuracy of uniform averaging with the lower cross-entropy of full 4:2:1 weighting.
change: Apply moderate spatial weights during translation-and-flip probability averaging and reuse the already-computed centered logits.
mechanism: Geometric-midpoint translation weighting
evidence_used: Uniform inference achieved 9,221 correct at 0.24150 cross-entropy, while stronger 4:2:1 weighting achieved 9,217 correct but improved cross-entropy to 0.23769; the geometric midpoint directly tests the tradeoff between them.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A per-axis 1:3:1 crop distribution will exceed 9,221 correct predictions by interpolating between the best 1:2:1 distribution and the slightly worse 1:4:1 distribution, while vectorized inference will prevent the prior test from timing out.
change: Change crop sampling from 1:2:1 to 1:3:1 and evaluate all 18 translated/flipped validation views in one forward pass.
mechanism: Vectorized intermediate center-biased translation augmentation
evidence_used: The 1:2:1 distribution achieved 9,221 correct versus 9,220 for 1:4:1, making 1:3:1 the unmeasured intermediate; its prior verification timed out, while vectorizing an even larger 26-view ensemble completed successfully.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting center, cardinal, and diagonal views by 2:√2:1 will exceed 9,221 correct predictions while vectorization avoids the previous timeout.
change: Vectorize the moderate-weight 18-view probability ensemble and reuse the already-computed centered logits.
mechanism: Vectorized geometric-midpoint translation ensemble
evidence_used: Uniform inference achieved 9,221 correct, while 4:2:1 weighting lowered cross-entropy but reduced correct predictions to 9,217; the unverified geometric midpoint targets that tradeoff, and a larger vectorized ensemble previously completed successfully.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reallocating diagonal crop probability to centered and cardinal crops will exceed 9,221 correct predictions by preserving useful translation robustness while reducing exposure to the least valuable shifts.
change: Replace independent per-axis 1:2:1 crop sampling with a joint distribution assigning probability 1/3 to the centered crop, 1/6 to each cardinal shift, and zero to diagonal shifts.
mechanism: Cross-shaped one-pixel crop augmentation
evidence_used: Cardinal translation inference improved validation_correct from 9,146 to 9,156, while adding all four diagonal views gained only one additional correct prediction; meanwhile, increasing centered training exposure improved the best result to 9,221.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 74.59527108306065, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.24330801086425782, "validation_score": 9232.402152962606}

RECENT RESULT
hypothesis: Raising centered-crop probability from 1/3 to 5/13 while retaining only cardinal translations will exceed 9,232 correct predictions by modestly reducing translation mismatch without the excessive center concentration of 1:4:1 sampling.
change: Change cross-shaped crop weights from 2:1 center-to-cardinal to 5:2, giving probability 5/13 to center and 2/13 to each cardinal shift.
mechanism: Moderately center-biased cross translation augmentation
evidence_used: Removing diagonal crops improved validation_correct from 9,221 to 9,232, while earlier moderate center weighting improved 9,203 to 9,221 and stronger 1:4:1 weighting fell slightly; this motivates a smaller center-weight increase within the successful cross-shaped distribution.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 71.07026554108597, "validation_accuracy": 0.9238, "validation_correct": 9238, "validation_cross_entropy": 0.24555059051513672, "validation_score": 9238.401428897234}

RECENT RESULT
hypothesis: Removing diagonal inference views will exceed 9,238 correct predictions by aligning evaluation with the successful cardinal-only crop distribution and avoiding weak out-of-distribution diagonal shifts.
change: Restrict validation averaging from the full 3×3 translation grid to the centered image and four one-pixel cardinal translations, retaining horizontal-flip averaging.
mechanism: Cardinal-only translation-and-flip ensemble
evidence_used: Removing diagonal crops from training improved validation_correct from 9,221 to 9,232, while adding diagonal inference views previously contributed only one correct prediction; the current cardinal-only training distribution therefore makes cardinal-only inference the most targeted next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 65.0580692498479, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.23624278297424317, "validation_score": 9249.404451299442}

RECENT RESULT
hypothesis: Increasing centered-crop probability from 5/13 to 3/7 while retaining only cardinal translations will exceed 9,249 correct predictions by continuing the demonstrated benefit of modestly reducing translation magnitude without reaching the previously harmful 4/9 center concentration.
change: Change cross-shaped crop weights from 5:2 center-to-cardinal to 3:1, giving probability 3/7 to the centered crop and 1/7 to each cardinal shift.
mechanism: Moderately stronger center-biased cross translation augmentation
evidence_used: Increasing the cardinal-only center weight from 2:1 to 5:2 improved validation_correct from 9,232 to 9,238, while removing diagonal inference views subsequently raised it to 9,249; 3:1 is the next controlled center-weight increase and remains below the unsuccessful 4/9 centered probability.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing centered-crop probability from 5/13 to 2/5 will exceed 9,249 correct predictions by continuing the observed benefit of modest center bias while remaining more conservative than the unverified 3/7 setting.
change: Change cardinal-only crop weights from 5:2 to 8:3, assigning probability 2/5 to the centered crop and 3/20 to each cardinal translation.
mechanism: Intermediate center-biased cross translation augmentation
evidence_used: Increasing center probability within cross-shaped augmentation from 1/3 to 5/13 improved validation_correct from 9,232 to 9,238; the next stronger 3/7 setting timed out, motivating this intermediate refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising centered-crop probability from 5/13 to 2/5 will exceed 9,249 correct predictions, while reusing centered inference logits will help the previously timed-out setting complete.
change: Change cardinal-only crop weights from 5:2 to 8:3 and eliminate the redundant second forward pass for the centered unflipped validation view.
mechanism: Runtime-efficient intermediate center-biased cross augmentation
evidence_used: Increasing center probability from 1/3 to 5/13 improved validation_correct from 9,232 to 9,238; the 2/5 experiment timed out without accuracy evidence, and the current evaluation unnecessarily computes centered logits twice.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging logits across the same cardinal and flipped views will exceed 9,249 correct predictions by rewarding cross-view class agreement instead of allowing one high-probability view to dominate the arithmetic probability ensemble.
change: Replace probability-space averaging with logit-space averaging while reusing the already-computed centered logits, preserving all ten inference views and the training procedure.
mechanism: Geometric-mean cardinal-view ensemble
evidence_used: Cardinal-only inference improved validation_correct from 9,238 to 9,249, confirming these views contain useful complementary evidence; spatial probability reweighting reduced accuracy, so changing only the aggregation rule is the most targeted remaining inference test.
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
