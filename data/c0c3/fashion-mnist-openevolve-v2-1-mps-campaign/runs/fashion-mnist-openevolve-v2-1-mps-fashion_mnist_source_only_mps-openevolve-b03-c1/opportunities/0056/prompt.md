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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 49.14878675015643, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24397592544555663, "validation_score": 9133.40193703895}
prior_hypothesis: Scaling ensemble logits by 1.034921875 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.243981142.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.03 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.245031295.
change: Increase the positive post-ensemble calibration scale from 1.02 to 1.03.
mechanism: Evaluation-only logit sharpening midpoint
evidence_used: Scaling from 1.00 to 1.02 preserved 9,133 correct predictions and reduced cross-entropy from 0.246640940 to 0.245031295, indicating residual underconfidence; the 1.04 verification timed out without contradicting that trend, so 1.03 is the most informative intermediate probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 57.836604417068884, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24431087265014648, "validation_score": 9133.401828844375}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.04 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.244310873.
change: Increase the positive post-ensemble calibration scale from 1.03 to 1.04.
mechanism: Evaluation-only logit sharpening continuation
evidence_used: Scaling from 1.02 to 1.03 preserved 9,133 correct predictions and reduced cross-entropy from 0.245031295 to 0.244310873; the prior 1.04 attempt timed out and therefore provides no contradictory performance evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.04 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.244310873.
change: Increase the positive post-ensemble calibration scale from 1.03 to 1.04.
mechanism: Evaluation-only logit sharpening continuation
evidence_used: Increasing the scale from 1.02 to 1.03 preserved 9,133 correct predictions and reduced cross-entropy from 0.245031295 to 0.244310873; the two 1.04 attempts timed out, which provides no contradictory performance evidence because this scalar operation does not materially change runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.15 will preserve all 9,133 argmax predictions while reducing validation cross-entropy below 0.244310873.
change: Increase the evaluation-only positive logit scale from 1.03 to 1.15.
mechanism: Quadratic-extrapolated logit calibration
evidence_used: Scales 1.00, 1.02, and 1.03 produced cross-entropies 0.246640940, 0.245031295, and 0.244310873 with identical correct counts; quadratic extrapolation places the calibration optimum near 1.153, while repeated 1.04 timeouts provide no performance contradiction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.035 will preserve all 9,133 argmax predictions while lowering validation cross-entropy below 0.244310873.
change: Increase the positive post-ensemble calibration scale from 1.03 to 1.035.
mechanism: Conservative evaluation-only logit sharpening
evidence_used: Increasing the scale from 1.02 to 1.03 preserved 9,133 correct predictions and reduced cross-entropy from 0.245031295 to 0.244310873; 1.035 is a conservative continuation of that measured trend after repeated 1.04 verifications timed out without producing contrary score evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 48.540404374944046, "validation_accuracy": 0.9132, "validation_correct": 9132, "validation_cross_entropy": 0.2439707160949707, "validation_score": 9132.401938722134}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.0325 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.244310873.
change: Increase the evaluation-only positive logit scale from 1.03 to the midpoint 1.0325.
mechanism: Post-ensemble calibration boundary bisection
evidence_used: Scale 1.03 achieved 9,133 correct at 0.244310873 cross-entropy, while 1.035 lowered cross-entropy to 0.243970716 but lost one correct; their midpoint is the most informative probe of the apparent float-sensitive accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 56.99472024990246, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24413915481567383, "validation_score": 9133.401884305356}

RECENT RESULT
hypothesis: Replacing hard max pooling with learned per-channel mixtures of max and average pooling will exceed 9,133 correct predictions by preserving both salient edges and regional occupancy across small translations.
change: Add a 144-parameter mixed-pooling module initialized to 75% max and 25% average, then use it at all three downsampling stages; total learned parameters become 245,178.
mechanism: Channel-adaptive mixed-statistic downsampling
evidence_used: Translation augmentation and multi-view inference underpin the strongest design, indicating sensitivity to spatial phase, while logit calibration improved cross-entropy but remained capped at 9,133 correct. This challenges the old assumption that every channel should retain only its local maximum and instead learns whether extrema or distributed activation is more useful at each stage.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.03375 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.244139155.
change: Increase the evaluation-only positive logit scale from 1.0325 to the midpoint 1.03375.
mechanism: Post-ensemble calibration boundary bisection
evidence_used: Scale 1.0325 retained 9,133 correct with 0.244139155 cross-entropy, while 1.035 lowered cross-entropy but lost one correct; their midpoint is the most informative remaining probe of the accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 61.65893970802426, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.2440545166015625, "validation_score": 9133.401911647221}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.034375 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.244054517.
change: Increase the evaluation-only positive logit scale from 1.03375 to the midpoint between the successful 1.03375 and accuracy-losing 1.035 scales.
mechanism: Post-ensemble calibration boundary bisection
evidence_used: Scale 1.03375 achieved 9,133 correct with 0.244054517 cross-entropy, while 1.035 reduced cross-entropy to 0.243970716 but lost one correct; their midpoint is the most informative probe of the observed accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 60.94481783313677, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24401251525878906, "validation_score": 9133.401925216882}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.0346875 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.244012515.
change: Increase the evaluation-only positive logit scale from 1.034375 to the midpoint between the successful 1.034375 and accuracy-losing 1.035 scales.
mechanism: Post-ensemble calibration boundary bisection
evidence_used: Scale 1.034375 retained 9,133 correct with 0.244012515 cross-entropy, while 1.035 produced lower cross-entropy but lost one correct; their midpoint is the most informative remaining calibration probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 57.51280308305286, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24399158248901368, "validation_score": 9133.401931980117}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.03484375 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.243991582.
change: Increase the evaluation-only positive logit scale from 1.0346875 to the midpoint between the successful 1.0346875 and accuracy-losing 1.035 scales.
mechanism: Post-ensemble calibration boundary bisection
evidence_used: Scale 1.0346875 retained 9,133 correct with 0.243991582 cross-entropy, while 1.035 achieved lower cross-entropy but lost one correct; their midpoint is the most informative remaining probe of the observed accuracy boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 56.27409670781344, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24398114242553712, "validation_score": 9133.401935353317}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.034921875 will retain 9,133 correct predictions while lowering validation cross-entropy below 0.243981142.
change: Increase the evaluation-only positive logit scale to the midpoint between the successful 1.03484375 and accuracy-losing 1.035 scales.
mechanism: Post-ensemble calibration boundary bisection
evidence_used: Scale 1.03484375 retained 9,133 correct with 0.243981142 cross-entropy, while 1.035 lowered cross-entropy but lost one correct; their midpoint is the most informative remaining calibration probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 49.14878675015643, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.24397592544555663, "validation_score": 9133.40193703895}



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
