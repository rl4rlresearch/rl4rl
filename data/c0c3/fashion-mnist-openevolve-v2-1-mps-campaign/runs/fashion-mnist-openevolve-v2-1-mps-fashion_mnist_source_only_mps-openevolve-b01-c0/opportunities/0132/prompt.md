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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.00727504189126, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}
prior_hypothesis: A 50.687969970703125% live / 49.312030029296875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969650268556.

## Recent verification evidence

RECENT RESULT
hypothesis: A 50.68797073364258% live / 49.31202926635742% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the nearest validated higher-live mixture.
mechanism: Upper-side sub-quantization live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 weight is best; the nearest tested mixtures on both sides were worse, but the higher-live neighbor produced lower cross-entropy than the lower-live neighbor, motivating refinement on the upper side.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 68.08024295885116, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969955444335, "validation_score": 9290.41580397752}

RECENT RESULT
hypothesis: A 50.68797035217285% live / 49.31202964782715% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the ensemble weight halfway from the current best mixture toward the nearest validated higher-live mixture.
mechanism: Upper-side eighth-step live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 live weight is best; the nearest tested mixtures on both sides were worse, but the higher-live neighbor had lower cross-entropy than the lower-live neighbor, making this untested upper midpoint the most informative refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 70.97980979201384, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}

RECENT RESULT
hypothesis: A 50.68797016143799% live / 49.31202983856201% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the live-model ensemble weight halfway from the current best toward the nearest tested higher-live mixture that tied its cross-entropy.
mechanism: Upper-side sixteenth-step live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 weight and the higher 0.5068797035217285 weight both achieved 9,290 correct with 0.20248969497680663 cross-entropy, while farther upper and lower probes were worse; their untested midpoint is the most informative remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.90004099998623, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}

RECENT RESULT
hypothesis: A 50.68797006607056% live / 49.31202993392944% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Move the live-model ensemble weight halfway from the current best toward the nearest higher-live mixture that tied its cross-entropy.
mechanism: Upper-side thirty-second-step live/EMA mixture refinement
evidence_used: The current 0.50687969970703125 weight and the tested 0.5068797016143799 weight both achieved 9,290 correct with 0.20248969497680663 cross-entropy, while farther probes were worse; their untested midpoint is the most informative remaining local refinement.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 72.49656666698866, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}

RECENT RESULT
hypothesis: Increasing the inference temperature by one float32 representable step will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.20248969497680663.
change: Raise the positive inference temperature from 0.717143714427948 to its next float32 value, 0.7171437740325928.
mechanism: One-ULP temperature calibration probe
evidence_used: Multiple increasingly fine live/EMA-weight probes tied or worsened the current cross-entropy, indicating that coordinate is quantization-limited; temperature is an independent calibration coordinate whose positive rescaling preserves prediction ordering.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the inference temperature to the next float32 value will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Increase the inference temperature from 0.717143714427948 to 0.7171437740325928.
mechanism: One-ULP temperature calibration retry
evidence_used: Fine live/EMA mixture probes only tied or worsened cross-entropy, while the sole independent one-ULP temperature probe timed out without metric evidence; retrying it is the most informative unresolved change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 58.10279475012794, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969802856445, "validation_score": 9290.415803978047}

RECENT RESULT
hypothesis: Lowering the inference temperature by one float32 representable step will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.20248969497680663.
change: Decrease the positive inference temperature from 0.717143714427948 to 0.7171436548233032.
mechanism: Opposite-direction one-ULP temperature calibration probe
evidence_used: Increasing the temperature by one ULP preserved 9,290 correct predictions but worsened cross-entropy to 0.20248969802856445, making the symmetric lower-temperature probe the most informative remaining calibration change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the inference temperature by one float32 ULP will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20248969497680663.
change: Decrease the positive inference temperature from 0.717143714427948 to 0.7171436548233032.
mechanism: Opposite-direction one-ULP temperature calibration retry
evidence_used: Raising temperature by one ULP preserved 9,290 correct but worsened cross-entropy to 0.20248969802856445; the symmetric lower-temperature probe timed out without metric evidence, so retrying it is the most informative unresolved change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 65.51739641604945, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2024896999359131, "validation_score": 9290.415803977387}

RECENT RESULT
hypothesis: Multiplying by the nearest float32 reciprocal will preserve all 9,290 predictions while reducing cross-entropy below 0.20248969497680663 by realizing an effective temperature slightly above the current value but below its next float32 value.
change: Replace temperature division with mathematically equivalent reciprocal multiplication, whose float32 rounding provides a finer calibration point.
mechanism: Reciprocal-multiply sub-ULP temperature interpolation
evidence_used: Both adjacent temperature probes worsened cross-entropy, but the higher-temperature probe worsened less (0.20248969802856445 versus 0.2024896999359131), suggesting the local optimum lies slightly above the current temperature within one ULP.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Multiplying logits by the nearest float32 reciprocal will retain all 9,290 correct predictions while reducing validation cross-entropy below 0.20248969497680663.
change: Replace inference-temperature division with equivalent reciprocal multiplication, producing a finer effective calibration point.
mechanism: Reciprocal-multiply sub-ULP temperature interpolation retry
evidence_used: Both adjacent float32 temperatures worsened cross-entropy, while the reciprocal-multiplication probe timed out without metric evidence; retrying this unresolved sub-ULP point is the most informative next change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 51.942605749936774, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.20248969497680663, "validation_score": 9290.415803979102}

RECENT RESULT
hypothesis: Increasing the unshifted view’s ensemble weight from 20% to 27.27% will correct at least one shift-induced prediction error and exceed 9,290 validation-correct predictions.
change: Weight the original and horizontally flipped original views 1.5× while retaining all shifted views, the live/EMA mixture, and temperature calibration.
mechanism: Canonical-view-weighted test-time augmentation
evidence_used: Adjacent live/EMA weights and temperatures only tied or worsened the 9,290-correct result, indicating those calibration coordinates are saturated; view weighting is an independent prediction coordinate, and the unshifted views preserve the validation image geometry without synthetic translation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 61.44102166686207, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20231712341308594, "validation_score": 9284.415863660479}

RECENT RESULT
hypothesis: Halving the unshifted view’s weight from 1.0 to 0.5 will correct at least one shift-sensitive error and exceed 9,290 validation-correct predictions.
change: Reduce the original and horizontally flipped original views from 20% to 11.11% of the TTA ensemble while preserving all shifted views, live/EMA mixing, and temperature calibration.
mechanism: Opposite-direction canonical-view downweighting
evidence_used: Increasing canonical-view weight to 1.5 reduced validation-correct from 9,290 to 9,284; probing the opposite direction is the most informative remaining view-weight change.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 50.404451332986355, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20278972625732422, "validation_score": 9289.415700258394}



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
