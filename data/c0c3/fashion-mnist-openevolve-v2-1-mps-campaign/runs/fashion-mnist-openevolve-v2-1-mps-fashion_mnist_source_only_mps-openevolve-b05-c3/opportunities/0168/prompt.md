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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 69.44678320805542, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.225365966796875, "validation_score": 9209.4080413636}
prior_hypothesis: Replacing one horizontally redundant shift pair with centered views while preserving the best verified 1:8 centered-to-shifted weight ratio will reach at least 9,208 correct predictions without increasing inference work.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 63.683481083018705, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.2237608757019043, "validation_score": 9210.408576552763}
prior_hypothesis: Restoring the verified 1/3 terminal centered-crop rate will recover 9,210 correct predictions, while 1.15× final sharpening preserves those argmax decisions and lowers cross-entropy below 0.2253669.

## Recent verification evidence

RECENT RESULT
hypothesis: The qualified half-weight centered ensemble will reproduce 9,208 correct predictions, exceeding the current 9,206, while 1.10× final sharpening reduces cross-entropy.
change: Give each centered view 0.5 weight, retain unit-weight shifted views, normalize by 9.0, and increase final sharpening from 1.05× to 1.10×.
mechanism: Half-weight centered-view calibrated probability TTA
evidence_used: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy; later verification failures supplied no contrary performance result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A centered-view weight of 0.375 will exceed 9,208 correct predictions by refining the observed interior optimum between the inferior zero- and unit-weight endpoints, while paired inference will avoid the prior verification timeout.
change: Batch each view with its horizontal flip in one forward pass, reduce both centered-view weights from 0.5 to 0.375, and normalize the unchanged eight unit-weight shifted views by the resulting total weight of 8.75.
mechanism: Pair-batched interior-weight probability TTA
evidence_used: Centered weights 0.0, 0.5, and 1.0 produced 9,206, 9,208, and 9,206 correct respectively; the previous 0.375 test timed out without performance evidence, so compute-equivalent paired inference makes that unresolved interpolation test more likely to complete.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Half-weighting both centered views will reproduce the qualified 9,208-correct result, exceeding the current 9,206 correct predictions.
change: Give each centered view 0.5 weight, retain unit weight for the eight shifted views, and normalize by total weight 9.0.
mechanism: Shift-favored probability TTA
evidence_used: Reference Design 1 verified this exact aggregation at 9,208 correct and 0.224548 cross-entropy; subsequent verification failures reported no contrary accuracy result, while the zero- and unit-weight endpoints each achieved only 9,206 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing one horizontally redundant shift pair with centered views while preserving the best verified 1:8 centered-to-shifted weight ratio will reach at least 9,208 correct predictions without increasing inference work.
change: Evaluate centered and flipped-centered views at 0.375 weight each, retain three unit-weight shift pairs, omit one horizontal shift pair, and normalize the eight-pass ensemble by 6.75.
mechanism: Flip-symmetric eight-pass centered TTA
evidence_used: Half-weight centered views with eight shifted views achieved the best verified result of 9,208 correct at a 1:8 centered-to-shifted weight ratio, whereas shift-only inference achieved 9,206. Prior ten-view attempts repeatedly failed verification, motivating the same ratio within the current eight-forward runtime budget.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 69.44678320805542, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.225365966796875, "validation_score": 9209.4080413636}

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
