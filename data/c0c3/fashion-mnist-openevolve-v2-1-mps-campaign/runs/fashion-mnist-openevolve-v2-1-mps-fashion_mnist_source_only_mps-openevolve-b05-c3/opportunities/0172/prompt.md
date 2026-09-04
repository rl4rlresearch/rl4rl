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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.83847858314402, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22708330039978028, "validation_score": 9206.40747029956}
prior_hypothesis: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 64.58129475009628, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2253669174194336, "validation_score": 9210.408041047047}
prior_hypothesis: Increasing terminal centered-crop frequency from 1/5 to 1/3 will exceed 9,209 correct predictions by reversing the centered-exposure reduction that dropped correctness to 9,193.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 66.18983545806259, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.2229884365081787, "validation_score": 9209.408834609612}
prior_hypothesis: Restoring the qualified 1/3-centered training sampler and increasing per-view sharpening from 1.10× to 1.15× will exceed 9,210 correct predictions by further emphasizing views with decisive class margins, while retaining the best verified 1.15× final calibration.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 62.26942599983886, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22300627403259277, "validation_score": 9210.408828646767}
prior_hypothesis: Increasing per-view logit sharpening from 1.05× to 1.10× will exceed 9,210 correct predictions by giving the most confident centered and shifted views greater influence without changing the qualified training trajectory or inference workload.

## Recent verification evidence

RECENT RESULT
hypothesis: Fusing aligned shallow, intermediate, and deep feature maps before classification will exceed 9,209 correct predictions by recovering fine-scale cues discarded by the shared deepest-only 3×3 representation.
change: Replace the deepest-only head with a parameter-budgeted learned pyramid fusion head, and retain the best verified eight-pass centered/shifted TTA aggregation.
mechanism: Learned multi-scale spatial-pyramid fusion
evidence_used: Reference Design 2 reached 9,209 correct, while extensive TTA-weight refinements remained within 9,206–9,209; every qualified design still classified only the final feature map, making cross-scale representation the clearest untested bottleneck. The fusion head remains lightweight, unlike the dynamic-channel design that timed out.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the final multiplier from 1.10× to 1.15× will preserve all 9,209 argmax predictions while reducing validation cross-entropy below 0.225366.
change: Sharpen the verified eight-pass ensemble’s final log-probabilities without changing training, parameters, inference views, or runtime materially.
mechanism: Runtime-neutral post-ensemble temperature sharpening
evidence_used: Raising final sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and reduced cross-entropy from 0.227083 to 0.224459; prior 1.15× attempts used slower ten-view ensembles and timed out, whereas the current eight-pass design verified in 69.45 seconds.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing terminal centered-crop frequency from 1/5 to 1/3 will exceed 9,209 correct predictions by reversing the centered-exposure reduction that dropped correctness to 9,193.
change: Restore the best verified eight-pass ensemble, then add a second centered outcome to the terminal crop sampler while retaining one outcome per cardinal shift.
mechanism: Center-biased terminal fine-tuning with qualified eight-view TTA
evidence_used: Reference Design 1 achieved 9,209 correct with 1/5 centered terminal sampling; reducing centered frequency to 1/9 produced only 9,193 correct, directly motivating a center-biased terminal phase.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 64.58129475009628, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2253669174194336, "validation_score": 9210.408041047047}

RECENT RESULT
hypothesis: Increasing terminal centered-crop frequency from 1/3 to 3/7 will exceed 9,210 correct predictions while retaining the verified runtime-safe ensemble.
change: Restore the best eight-pass centered/shifted aggregation and give terminal sampling three centered outcomes versus four cardinal-shift outcomes.
mechanism: Center-frequency continuation with qualified eight-view TTA
evidence_used: The qualified eight-view design improved from 9,209 correct at 1/5 centered sampling to 9,210 at 1/3, while reducing centered exposure to 1/9 produced only 9,193; 3/7 tests whether that observed trend continues without changing model capacity.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 77.07978883292526, "validation_accuracy": 0.9208, "validation_correct": 9208, "validation_cross_entropy": 0.22575960273742676, "validation_score": 9208.407910326694}

RECENT RESULT
hypothesis: A 30% terminal centered-crop rate will exceed 9,210 correct predictions by targeting the apparent optimum between 20% (9,209), 33.3% (9,210), and 42.9% (9,208), while retaining the best verified eight-view ensemble.
change: Restore the qualified eight-pass centered/shifted TTA and replace terminal crop sampling with 12 centered and seven outcomes per cardinal shift out of 40 equally likely outcomes.
mechanism: Interior terminal center-frequency interpolation
evidence_used: Correctness rose from 9,209 at 20% centered exposure to 9,210 at 33.3%, then fell to 9,208 at 42.9%; these observations motivate testing a balanced 30% interior setting without changing model capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 61.411563417175785, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22618416595458984, "validation_score": 9192.407769088757}

RECENT RESULT
hypothesis: Raising each centered TTA view from 0.375 to 0.5 weight will exceed 9,210 correct predictions because the best model now receives substantially more centered terminal training exposure.
change: Increase both centered-view ensemble weights to 0.5 and renormalize the unchanged six shifted-view weights by the new total weight of 7.0.
mechanism: Training-aligned centered-view reweighting
evidence_used: Increasing terminal centered sampling from 20% to 33.3% improved correctness from 9,209 to 9,210, while prior TTA evidence found half-weight centered views superior to zero- and unit-weight endpoints; this motivates aligning inference more strongly with centered training.
result: training did not finish within the verification time limit

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
