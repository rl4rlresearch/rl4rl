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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 67.10024604201317, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22289843368530274, "validation_score": 9210.408864699002}
prior_hypothesis: A 1.178× final multiplier will preserve all 9,210 predictions while reducing validation cross-entropy below 0.22289956.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.64136466709897, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2229105945587158, "validation_score": 9210.408860633168}
prior_hypothesis: Restoring 1.10× per-view sharpening will recover exactly 9,210 correct predictions, while a 1.1875× argmax-invariant final multiplier will reduce cross-entropy below 0.2228996.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 62.26942599983886, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22300627403259277, "validation_score": 9210.408828646767}
prior_hypothesis: Increasing per-view logit sharpening from 1.05× to 1.10× will exceed 9,210 correct predictions by giving the most confident centered and shifted views greater influence without changing the qualified training trajectory or inference workload.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}
prior_hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the final log-probability multiplier from 1.15× to 1.20× will preserve all 9,210 argmax predictions while reducing validation cross-entropy below 0.2230063.
change: Sharpen only the final eight-view ensemble output, leaving training, per-view weighting, parameters, and inference workload unchanged.
mechanism: Post-ensemble temperature calibration
evidence_used: Raising final sharpening from 1.10× to 1.15× preserved 9,210 correct predictions and reduced cross-entropy from 0.2253669 to 0.2237609; the prior 1.20× attempt only timed out, and positive final scaling is argmax-invariant.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The qualified 1.10× per-view ensemble will retain exactly 9,210 correct predictions, while increasing its argmax-invariant final multiplier from 1.15× to 1.20× will reduce validation cross-entropy below 0.2230063.
change: Restore the best verified 1.10× per-view sharpening and apply 1.20× final log-probability sharpening without changing training, parameters, or inference workload.
mechanism: Confidence-weighted TTA with stronger post-ensemble calibration
evidence_used: Reference Design 2 achieved 9,210 correct with 0.2230063 cross-entropy at 1.10× per-view and 1.15× final sharpening; earlier raising final sharpening from 1.10× to 1.15× preserved 9,210 correct and lowered cross-entropy, while the 1.20× attempts timed out without accuracy evidence against the change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 1.125× per-view multiplier will restore 9,210 correct predictions while lowering cross-entropy below the 0.2230063 achieved at 1.10×.
change: Replace the current 1.15× per-view sharpening with the unverified 1.125× midpoint, retaining the qualified 1/3-centered training sampler and 1.15× final calibration.
mechanism: Boundary-seeking per-view temperature interpolation
evidence_used: Per-view sharpening retained 9,210 correct at 1.10× but lost one prediction at 1.15×; the prior 1.125× attempt timed out, so the midpoint remains the most direct unresolved boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.27186129195616, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.2229149429321289, "validation_score": 9209.408859179364}

RECENT RESULT
hypothesis: Restoring the verified 1.10× per-view ensemble will recover 9,210 correct predictions, while increasing only its final multiplier to 1.20× will lower cross-entropy below 0.2230063 without changing those predictions.
change: Restore the qualified 1/3-centered terminal sampler and weighted eight-view TTA, then apply 1.20× post-ensemble sharpening.
mechanism: Qualified confidence-weighted TTA with argmax-invariant calibration
evidence_used: Reference Design 3 achieved 9,210 correct with 0.2230063 cross-entropy at 1.10× per-view and 1.15× final sharpening; prior final-sharpening increases preserved argmax decisions and lowered cross-entropy, while 1.20× remains unresolved only because verification timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the final multiplier to 1.20× will preserve all 9,210 argmax predictions while reducing validation cross-entropy below 0.2230063.
change: Increase only the argmax-invariant final ensemble multiplier from 1.15× to 1.20×.
mechanism: Post-ensemble temperature sharpening
evidence_used: Raising final sharpening from 1.10× to 1.15× previously retained 9,210 correct predictions and reduced cross-entropy; 1.20× remains unresolved only because its verifications timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring 1.10× per-view sharpening will retain 9,210 correct predictions, while a 1.175× argmax-invariant final multiplier will reduce cross-entropy below 0.2230063.
change: Increase per-view sharpening from 1.05× to the best verified 1.10× and test final sharpening midway between the verified 1.15× setting and repeatedly timed-out 1.20× setting.
mechanism: Qualified confidence-weighted TTA with midpoint post-ensemble calibration
evidence_used: Reference Design 2 achieved 9,210 correct and 0.2230063 cross-entropy with 1.10× per-view and 1.15× final sharpening; prior increases in final sharpening preserved argmax predictions while lowering cross-entropy.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 74.83207758283243, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22289955978393555, "validation_score": 9210.408864322502}

RECENT RESULT
hypothesis: Restoring 1.10× per-view sharpening will recover exactly 9,210 correct predictions, while a 1.1875× argmax-invariant final multiplier will reduce cross-entropy below 0.2228996.
change: Restore the best verified per-view temperature and increase final ensemble sharpening halfway from the qualified 1.175× setting toward the unresolved 1.20× setting.
mechanism: Qualified confidence-weighted TTA with finer post-ensemble calibration
evidence_used: The 1.10× per-view, 1.175× final design achieved the best verified score with 9,210 correct and 0.2228996 cross-entropy; 1.125× per-view lost one correct prediction, while positive final scaling preserves argmax decisions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.64136466709897, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2229105945587158, "validation_score": 9210.408860633168}

RECENT RESULT
hypothesis: Exposing global summaries of the first two residual stages directly to the classifier will exceed 9,210 correct predictions by preserving fine texture and mid-level shape evidence that the final 3×3 representation may discard.
change: Replace the final-stage-only head with a parameter-budget-neutral 245,571-parameter multi-scale head, while restoring the best-qualified training sampler and TTA controls.
mechanism: Hierarchical multi-scale feature fusion
evidence_used: Reference Design 1 reached 9,210 correct, but every qualified design used the same final-stage-only representation and further TTA temperature tuning plateaued or regressed. This challenges that load-bearing architectural assumption through direct learned fusion of early, middle, and final features.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245571, "training_seconds": 64.44317704206333, "validation_accuracy": 0.9193, "validation_correct": 9193, "validation_cross_entropy": 0.22412265777587892, "validation_score": 9193.408455800425}

RECENT RESULT
hypothesis: A 1.178× final multiplier will preserve all 9,210 predictions while reducing validation cross-entropy below 0.22289956.
change: Increase only the argmax-invariant final ensemble multiplier from 1.175× to 1.178×.
mechanism: Bracketed post-ensemble temperature calibration
evidence_used: Cross-entropy improved from 0.22300627 at 1.15× to 0.22289956 at 1.175×, then worsened to 0.22291059 at 1.1875×; interpolation places the calibration minimum near 1.178×.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 67.10024604201317, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22289843368530274, "validation_score": 9210.408864699002}

RECENT RESULT
hypothesis: A 1.17792× final multiplier will preserve all 9,210 argmax predictions while reducing validation cross-entropy below 0.2228984337.
change: Replace the current 1.1875× final calibration with the estimated minimum of the bracketed cross-entropy curve.
mechanism: Convex post-ensemble temperature refinement
evidence_used: Cross-entropy was 0.2228995598 at 1.175×, improved to 0.2228984337 at 1.178×, and worsened to 0.2229105946 at 1.1875×; local quadratic interpolation places the minimum near 1.17792×, and positive scaling cannot change argmax predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying the verified 1.178× final multiplier will preserve all 9,210 argmax predictions while reducing validation cross-entropy from 0.22300627 to approximately 0.22289843.
change: Increase only the argmax-invariant final ensemble multiplier from 1.15× to the best verified 1.178× setting.
mechanism: Verified post-ensemble temperature calibration
evidence_used: Reference Design 1 achieved the highest available validation_score, retaining 9,210 correct predictions while improving cross-entropy to 0.2228984337; nearby 1.175× and 1.1875× settings were both worse.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact best-qualified inference ensemble and terminal sampler will recover 9,210 correct predictions with validation cross-entropy near 0.22289843.
change: Replace equal-weight ten-view TTA with the verified eight-pass weighted ensemble using 1.10× per-view and 1.178× final sharpening, and restore the terminal sampler’s one-third centered-crop probability.
mechanism: Qualified confidence-weighted eight-view TTA with calibrated post-ensemble sharpening
evidence_used: Reference Design 1 achieved the highest verified score, 9,210 correct with 0.2228984337 cross-entropy; the current design’s altered sampler and ensemble achieved only 9,206 correct.
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
