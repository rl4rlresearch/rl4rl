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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 65.8712894169148, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.22187443466186524, "validation_score": 9256.409207350458}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.12 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2243048.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 52.59167395881377, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2243048080444336, "validation_score": 9256.408395030972}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.09 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2272195.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 62.20644187484868, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.22721946601867676, "validation_score": 9256.407425088866}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.06 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2306832.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.88835083297454, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21824926071166992, "validation_score": 9256.41042503872}
prior_hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.18 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2198722.

## Recent verification evidence

RECENT RESULT
hypothesis: A 3.75:1 center-to-shift inference weight will exceed 9,253 correct predictions by extending the verified improvement through 3.5:1 while probing below the inconclusive 4:1 setting.
change: Increase center-view weight from 2.0 to 3.75 and normalize the resulting probability mixture by 15.5.
mechanism: Fine-grained center-emphasized test-time augmentation
evidence_used: Increasing center weight from 3:1 to 3.5:1 retained 9,253 correct while lowering cross-entropy from 0.2350803 to 0.2344862; the 4:1 verification timed out without accuracy evidence, motivating a 3.75:1 intermediate test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 78.46530316607095, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.23422339134216308, "validation_score": 9250.405113048017}

RECENT RESULT
hypothesis: A 3.25:1 center-to-shift inference weight will exceed 9,253 correct predictions by targeting the quadratic correctness peak implied by the verified 3:1, 3.5:1, and 3.75:1 results.
change: Weight both center-view predictions by 3.25 while retaining unit weight for shifted views, and normalize the probability mixture by 14.5.
mechanism: Quadratic center-weight interpolation
evidence_used: The 3:1 and 3.5:1 designs each achieved 9,253 correct, while 3.75:1 fell to 9,250; quadratic interpolation of these results places the estimated correctness peak at 3.25:1.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 69.25696683302522, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2347708953857422, "validation_score": 9256.404933418717}

RECENT RESULT
hypothesis: A 3.30:1 center-to-shift inference weight with the verified 2.2% cosine floor will retain at least 9,256 correct predictions while lowering cross-entropy below 0.2347709.
change: Restore the best 2.2% learning-rate floor and use 3.30 weight for each center prediction versus unit weight for shifted predictions, normalized by 14.6.
mechanism: High-side center-weight refinement
evidence_used: The 3.25:1 design achieved the best 9,256 correct; increasing to 3.5:1 reduced cross-entropy from 0.2347709 to 0.2344862 but lost three correct predictions, motivating a small high-side step that may preserve correctness while capturing some cross-entropy improvement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 73.02317087491974, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.23471206130981445, "validation_score": 9255.404952713809}

RECENT RESULT
hypothesis: A 3.20:1 center-to-shift inference weight will exceed 9,256 correct predictions by probing below the current peak after 3.30:1 lost one correct prediction.
change: Reduce each center-view weight from 3.25 to 3.20 and normalize the probability mixture by 14.4.
mechanism: Low-side center-weight refinement
evidence_used: The 3.25:1 design achieved 9,256 correct, while 3.30:1 achieved 9,255 and 3.00:1 achieved 9,253; a symmetric low-side step is the most informative local refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 3.275:1 center-to-shift inference weight will retain the 9,256 correct predictions achieved at 3.25:1 while lowering cross-entropy below 0.2347709.
change: Reduce each center-view weight from 3.30 to 3.275 and normalize the probability mixture by 14.55.
mechanism: Midpoint center-weight refinement
evidence_used: The verified 3.25:1 design achieved the best 9,256 correct, while 3.30:1 lost one correct but lowered cross-entropy; their midpoint is the most informative untested tradeoff.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 72.45098183304071, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.23474135818481445, "validation_score": 9255.40494310544}

RECENT RESULT
hypothesis: Restoring the verified 3.25:1 center weight will recover 9,256 correct predictions, while multiplying the aggregated logits by 1.03 will preserve every argmax and reduce cross-entropy below 0.2347709.
change: Restore the best verified center-view weighting and apply mild post-ensemble sharpening to the returned logits.
mechanism: Argmax-preserving ensemble temperature sharpening
evidence_used: The 3.25:1 design achieved the best result with 9,256 correct; nearby higher weights reduced cross-entropy but lost predictions. Positive logit scaling leaves those 3.25:1 predictions unchanged while targeting the tie-breaking metric.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified 3.25:1 center weighting will recover 9,256 correct predictions, while scaling its aggregate logits by 1.03 will preserve those argmaxes and reduce cross-entropy below 0.2347709.
change: Restore the best center-view weight and apply mild post-ensemble logit sharpening.
mechanism: Argmax-preserving ensemble temperature sharpening
evidence_used: The 3.25:1 design achieved the best verified correctness at 9,256; the previous 1.03 sharpening attempt only timed out, providing no contrary accuracy or cross-entropy evidence, and positive scaling cannot alter its predicted classes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 60.55019558290951, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.23068321075439452, "validation_score": 9256.40627839531}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.06 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2306832.
change: Increase the post-ensemble logit multiplier from 1.03 to 1.06, leaving training and inference weighting unchanged.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Scaling from 1.00 to 1.03 preserved 9,256 correct predictions and reduced cross-entropy from 0.2347709 to 0.2306832; another equal-sized step tests whether the ensemble remains underconfident.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 62.20644187484868, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.22721946601867676, "validation_score": 9256.407425088866}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.09 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2272195.
change: Apply a 1.09 multiplier to the aggregate inference logits while leaving training and ensemble weighting unchanged.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Increasing the multiplier from 1.03 to 1.06 preserved 9,256 correct predictions and reduced cross-entropy from 0.2306832 to 0.2272195; another equal step tests the continuing underconfidence trend without changing argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 52.59167395881377, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2243048080444336, "validation_score": 9256.408395030972}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.12 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2243048.
change: Restore the best center-view weighting and increase the post-ensemble logit multiplier from 1.09 to 1.12.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Multipliers of 1.03, 1.06, and 1.09 each preserved 9,256 correct predictions while progressively reducing cross-entropy to 0.2306832, 0.2272195, and 0.2243048; the continuing improvement motivates another equal-sized sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 65.8712894169148, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.22187443466186524, "validation_score": 9256.409207350458}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.15 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2218744.
change: Restore the best center-view weighting and increase the post-ensemble logit multiplier from 1.12 to 1.15.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Multipliers of 1.03, 1.06, 1.09, and 1.12 all preserved 9,256 correct predictions while progressively lowering cross-entropy to 0.2306832, 0.2272195, 0.2243048, and 0.2218744, motivating one more equal sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 60.37882804102264, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.2198721591949463, "validation_score": 9256.40987901579}

RECENT RESULT
hypothesis: Scaling the verified 3.25:1 ensemble logits by 1.18 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2198722.
change: Increase the post-ensemble logit multiplier from 1.15 to 1.18, leaving training and ensemble weighting unchanged.
mechanism: Incremental argmax-preserving temperature sharpening
evidence_used: Multipliers from 1.03 through 1.15 consistently preserved 9,256 correct predictions while progressively lowering cross-entropy; the latest 1.15 result reached 0.2198722, supporting another equal sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 50.88835083297454, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.21824926071166992, "validation_score": 9256.41042503872}



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
