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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 64.58129475009628, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2253669174194336, "validation_score": 9210.408041047047}
prior_hypothesis: Increasing terminal centered-crop frequency from 1/5 to 1/3 will exceed 9,209 correct predictions by reversing the centered-exposure reduction that dropped correctness to 9,193.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 72.27186129195616, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.2229149429321289, "validation_score": 9209.408859179364}
prior_hypothesis: A 1.125× per-view multiplier will restore 9,210 correct predictions while lowering cross-entropy below the 0.2230063 achieved at 1.10×.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 62.26942599983886, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22300627403259277, "validation_score": 9210.408828646767}
prior_hypothesis: Increasing per-view logit sharpening from 1.05× to 1.10× will exceed 9,210 correct predictions by giving the most confident centered and shifted views greater influence without changing the qualified training trajectory or inference workload.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}
prior_hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the verified 1/3 terminal centered-crop rate will reproduce 9,210 correct predictions, improving on the current 9,209-correct implementation.
change: Increase terminal crop outcomes from five to six, adding a second centered outcome while preserving one outcome for each cardinal shift.
mechanism: Center-biased terminal crop sampling
evidence_used: Reference Design 1 verified this exact change at 9,210 correct and 0.2253669 cross-entropy, the best qualified result; the current 1/5 centered rate reached 9,209, while increasing further to 3/7 fell to 9,208.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the verified 1/3 terminal centered-crop rate will recover 9,210 correct predictions, while 1.15× final sharpening preserves those argmax decisions and lowers cross-entropy below 0.2253669.
change: Sample two centered and four cardinal-shift outcomes during terminal training, then increase the eight-view ensemble’s final log-probability multiplier from 1.10× to 1.15×.
mechanism: Center-biased terminal fine-tuning with argmax-invariant ensemble sharpening
evidence_used: The exact 1/3 terminal sampler achieved the best qualified correctness of 9,210, whereas 3/7 fell to 9,208; increasing sharpening from 1.05× to 1.10× previously preserved correctness and reduced cross-entropy, while the runtime-neutral 1.15× test remains unresolved.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 63.683481083018705, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2237608757019043, "validation_score": 9210.408576552763}

RECENT RESULT
hypothesis: The qualified 1/3-centered eight-view design will retain 9,210 correct predictions while increasing final sharpening to 1.20× lowers validation cross-entropy below 0.2237609.
change: Restore the best qualified centered/shifted ensemble and terminal sampler, then increase its argmax-invariant final multiplier from 1.15× to 1.20×.
mechanism: Continued post-ensemble temperature sharpening
evidence_used: On the same 9,210-correct design, raising final sharpening from 1.10× to 1.15× reduced cross-entropy from 0.2253669 to 0.2237609; continued sharpening tests whether that verified calibration trend persists without changing predictions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing per-view logit sharpening from 1.05× to 1.10× will exceed 9,210 correct predictions by giving the most confident centered and shifted views greater influence without changing the qualified training trajectory or inference workload.
change: Increase only the pre-softmax temperature multiplier for every TTA view, retaining the verified 1/3-centered training sampler, view weights, and 1.15× final calibration.
mechanism: Stronger confidence-weighted probability TTA
evidence_used: The current 1.05× per-view ensemble is the best qualified 9,210-correct design, while prior evidence specifically identified per-view sharpening as the runtime-neutral mechanism capable of changing ensemble argmax decisions; unlike final sharpening, this isolates a correctness-relevant inference adjustment.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 62.26942599983886, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22300627403259277, "validation_score": 9210.408828646767}

RECENT RESULT
hypothesis: Sample-conditioned channel gates will exceed 9,210 correct predictions by adding global image context while retaining the qualified model’s initial representation and low runtime.
change: Replace the assumption that channel importance is image-independent with lightweight squeeze/excitation gates in every residual stage. Zero initialization makes each gate an exact identity initially; the resulting model has 249,618 learned parameters.
mechanism: Identity-initialized global channel recalibration
evidence_used: Crop and TTA refinements repeatedly plateaued at 9,206–9,210 correct while every qualified design retained identical fixed-channel residual stages, making image-conditioned feature selection a distinct untested mechanism.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the qualified 1/3-centered training sampler and increasing per-view sharpening from 1.10× to 1.15× will exceed 9,210 correct predictions by further emphasizing views with decisive class margins, while retaining the best verified 1.15× final calibration.
change: Use two centered outcomes and four cardinal shifts during terminal training, sharpen every centered and shifted TTA view by 1.15×, and apply the qualified 1.15× post-ensemble multiplier.
mechanism: Continued confidence-weighted probability TTA
evidence_used: With the same 1/3-centered trajectory, increasing per-view sharpening from 1.05× to 1.10× retained 9,210 correct while reducing cross-entropy from 0.2237609 to 0.2230063; continuing this correctness-relevant adjustment is the most direct test beyond the current plateau.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 66.18983545806259, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.2229884365081787, "validation_score": 9209.408834609612}

RECENT RESULT
hypothesis: A 1.125× per-view multiplier will preserve the 9,210 correct predictions achieved at 1.10× while reducing validation cross-entropy below 0.2230063, locating a better point before the one-prediction regression at 1.15×.
change: Restore the qualified 1/3-centered training sampler and eight-pass weighted TTA, use the midpoint 1.125× per-view sharpening, and retain 1.15× final calibration.
mechanism: Boundary-seeking per-view temperature interpolation
evidence_used: Reference Design 3 achieved 9,210 correct at 1.10× per-view sharpening, while 1.15× reduced cross-entropy slightly but lost one correct prediction; bisecting that interval is the most direct test for retaining correctness while improving the tie-breaker.
result: training did not finish within the verification time limit

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
