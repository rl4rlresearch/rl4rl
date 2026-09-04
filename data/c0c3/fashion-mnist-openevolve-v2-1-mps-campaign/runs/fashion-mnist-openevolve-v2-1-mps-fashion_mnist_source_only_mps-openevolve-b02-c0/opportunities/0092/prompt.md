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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 79.37466162489727, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2134489387512207, "validation_score": 9253.412048652426}
prior_hypothesis: Increasing the positive ensemble scale from 1.28 to 1.281 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2134692.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling the verified ensemble scores from 1.275 to 1.325 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.
change: Change only the positive inference-time calibration multiplier; architecture, training, and ensemble decisions remain unchanged.
mechanism: Convex post-ensemble temperature calibration
evidence_used: Successive scales from 1.10 through 1.275 preserved all 9,253 predictions while monotonically reducing cross-entropy, and the observed marginal improvements suggest the convex calibration optimum lies near 1.325; prior higher-scale timeouts provide no contrary score evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the geometric contribution from 25% to 30% will preserve at least 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.
change: Shift five percentage points of inference weight from arithmetic probability averaging to geometric log-probability averaging, leaving training and calibration unchanged.
mechanism: Conservative geometric-ensemble interpolation
evidence_used: The 25% geometric blend improved validation_correct from 9,252 to 9,253, while the fully geometric ensemble had lower cross-entropy but lost two predictions; a small increase tests whether more of its calibration benefit is available within the current accuracy plateau.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the geometric contribution from 25% to 27.5% will retain at least 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.
change: Shift 2.5 percentage points of inference weight from arithmetic probability averaging to geometric log-probability averaging, retaining the verified training procedure, ten views, and 1.275 calibration.
mechanism: Fine-grained arithmetic–geometric ensemble interpolation
evidence_used: The 25% blend achieved the best verified count of 9,253, while the fully geometric ensemble had lower cross-entropy but two fewer correct predictions; 27.5% conservatively probes the interval below the timed-out 30% attempt.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.275 to 1.28 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.
change: Change only the inference-time calibration multiplier, leaving training, views, and ensemble decisions unchanged.
mechanism: Fine-grained post-ensemble temperature calibration
evidence_used: Every verified increase from 1.10 through 1.275 preserved 9,253 correct predictions and monotonically lowered cross-entropy; positive scaling preserves argmax, while the later timeouts supplied no contrary calibration evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 73.9237881670706, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2134691780090332, "validation_score": 9253.412041779933}

RECENT RESULT
hypothesis: Scaling ensemble scores from 1.28 to 1.35 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2134692.
change: Change only the positive inference-time calibration multiplier; training, augmentation, views, and ensemble weights remain unchanged.
mechanism: Evidence-fitted post-ensemble temperature calibration
evidence_used: Every verified scale increase through 1.28 preserved 9,253 correct and reduced cross-entropy; the recent 1.275-to-1.28 improvement shows sharpening remains beneficial, while positive scaling preserves argmax and prior higher-scale timeouts provide no contrary score evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.28 to 1.285 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2134692.
change: Change only the inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.
mechanism: Fine-grained post-ensemble temperature calibration
evidence_used: Increasing the scale from 1.275 to 1.28 preserved all 9,253 predictions and reduced cross-entropy from 0.2135743 to 0.2134692. Positive scaling preserves argmax predictions, and higher-scale timeouts supplied no contrary accuracy or calibration evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.28 to 1.2825 will preserve exactly 9,253 predictions while reducing validation cross-entropy below 0.2134692.
change: Change only the inference-time calibration multiplier; retain the verified training, architecture, views, and ensemble weights.
mechanism: Fine-grained post-ensemble temperature calibration
evidence_used: The verified increase from 1.275 to 1.28 preserved all 9,253 correct predictions and lowered cross-entropy from 0.2135743 to 0.2134692; positive scaling preserves argmax, and the timed-out 1.285 run supplied no contrary score evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.28 to 1.281 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2134692.
change: Change only the inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.
mechanism: Conservative post-ensemble temperature sharpening
evidence_used: The verified increase from 1.275 to 1.28 preserved 9,253 correct predictions and lowered cross-entropy from 0.2135743 to 0.2134692. Positive scaling preserves argmax, and the timed-out higher-scale attempts provided no contrary validation evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 79.37466162489727, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2134489387512207, "validation_score": 9253.412048652426}

RECENT RESULT
hypothesis: Scaling ensemble scores from 1.281 to 1.36 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2134489.
change: Change only the positive inference-time calibration multiplier; training, architecture, views, and ensemble weights remain unchanged.
mechanism: Evidence-fitted convex temperature calibration
evidence_used: Verified scaling from 1.20 through 1.281 preserved exactly 9,253 correct predictions while monotonically reducing cross-entropy. Recent finite differences indicate the convex calibration optimum is near 1.36, while prior higher-scale timeouts supplied no contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the ensemble scale from 1.281 to 1.282 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21344894.
change: Change only the positive inference-time calibration multiplier from 1.281 to 1.282.
mechanism: Conservative post-ensemble temperature sharpening
evidence_used: The verified increase from 1.280 to 1.281 preserved exactly 9,253 correct predictions and reduced cross-entropy from 0.21346918 to 0.21344894; a positive scalar preserves argmax decisions, and the small increment conservatively tests the continuing calibration trend.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the ensemble scale from 1.281 to 1.28125 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21344894.
change: Change only the positive inference-time calibration multiplier, retaining the verified architecture, training procedure, views, and ensemble weights.
mechanism: Fine-grained post-ensemble temperature sharpening
evidence_used: The verified increase from 1.280 to 1.281 preserved exactly 9,253 correct predictions and reduced cross-entropy from 0.21346918 to 0.21344894. Positive scaling preserves argmax decisions, and the timed-out 1.282 attempt supplied no contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the ensemble scale from 1.281 to 1.2811 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21344894.
change: Change only the positive inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.
mechanism: Ultra-conservative post-ensemble temperature sharpening
evidence_used: The verified increase from 1.280 to 1.281 preserved exactly 9,253 correct predictions and lowered cross-entropy from 0.21346918 to 0.21344894. Positive scaling preserves argmax, while timed-out 1.28125 and 1.282 attempts supplied no contrary validation evidence.
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
