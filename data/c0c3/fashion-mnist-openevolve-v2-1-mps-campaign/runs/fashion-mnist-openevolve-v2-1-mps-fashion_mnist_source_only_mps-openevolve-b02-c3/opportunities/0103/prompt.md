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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.15807658410631, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18770753898620607, "validation_score": 9347.420979057208}
prior_hypothesis: Vertical weights of 1.31640625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.23315116693266, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877075828552246, "validation_score": 9348.420979041657}
prior_hypothesis: Increasing the vertical-shift preference slightly beyond the best 1.25/0.75 weighting will exceed 9,346 correct predictions or retain 9,346 while lowering cross-entropy below 0.18771290.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.36736466712318, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.187707568359375, "validation_score": 9348.420979046796}
prior_hypothesis: Vertical weights of 1.3134765625 will retain 9,348 correct predictions while lowering cross-entropy below 0.187707582855.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.86007670802064, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.1877075584411621, "validation_score": 9347.420979050312}
prior_hypothesis: Vertical weights of 1.314453125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.

## Recent verification evidence

RECENT RESULT
hypothesis: Midpoint weights of 1.34375 vertical and 0.65625 horizontal will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy below 0.18770758.
change: Restore the best TTA direction and test the midpoint between the best 1.3125/0.6875 weights and the regressive 1.375/0.625 weights, preserving center and total weight.
mechanism: Bracketed vertical-shift weight refinement
evidence_used: Vertical weighting improved from 9,346 correct at 1.25 to 9,348 at 1.3125, then declined to 9,347 at 1.375, directly bracketing a likely optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 71.79709354205988, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18770811920166017, "validation_score": 9347.42097885155}

RECENT RESULT
hypothesis: Vertical weights of 1.328125 will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy below 0.18770758.
change: Restore the best-verified 0.98 EMA and test the midpoint between the best 1.3125/0.6875 TTA weights and the regressive 1.34375/0.65625 weights.
mechanism: Bracketed vertical-shift TTA refinement
evidence_used: Correctness peaked at 9,348 with 1.3125 vertical weights and declined to 9,347 at both 1.34375 and 1.375, tightly bracketing the optimum above 1.3125.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 72.55939345806837, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18770758590698242, "validation_score": 9347.420979040577}

RECENT RESULT
hypothesis: Vertical weights of 1.3203125 will retain at least 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.
change: Test the midpoint between the best 1.3125/0.6875 weights and the narrowly regressive 1.328125/0.671875 weights, preserving center and total ensemble weight.
mechanism: Quadratic-bracketed vertical-shift TTA refinement
evidence_used: The 1.3125 design achieved 9,348 correct; 1.328125 lost one prediction but had nearly identical cross-entropy, while 1.34375 worsened cross-entropy more clearly. Their cross-entropy trend places the interpolated minimum near this midpoint.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 70.74573524994776, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18770751800537108, "validation_score": 9347.420979064644}

RECENT RESULT
hypothesis: Vertical weights of 1.31640625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.
change: Test the midpoint between the 9,348-correct 1.3125 weighting and the lower-cross-entropy but 9,347-correct 1.3203125 weighting, preserving center and total ensemble weight.
mechanism: Boundary-aware vertical-shift TTA bisection
evidence_used: Weight 1.3125 achieved 9,348 correct, while 1.3203125 achieved 9,347 with slightly lower cross-entropy; their midpoint directly probes the decision boundary for a correctness-preserving cross-entropy gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 73.15807658410631, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18770753898620607, "validation_score": 9347.420979057208}

RECENT RESULT
hypothesis: Vertical weights of 1.314453125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707582855.
change: Restore the favorable TTA region and test the midpoint between the 9,348-correct 1.3125 weighting and the 9,347-correct 1.31640625 weighting, preserving total ensemble weight.
mechanism: Boundary-aware vertical-shift TTA bisection
evidence_used: Weight 1.3125 achieved 9,348 correct, while 1.31640625 lost one prediction but reduced cross-entropy to 0.187707538986; their midpoint most directly locates the correctness boundary and tests for a tie-breaking cross-entropy gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.86007670802064, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.1877075584411621, "validation_score": 9347.420979050312}

RECENT RESULT
hypothesis: Vertical weights of 1.3134765625 will retain 9,348 correct predictions while lowering cross-entropy below 0.187707582855.
change: Test the midpoint between the 9,348-correct 1.3125 weighting and the lower-cross-entropy but 9,347-correct 1.314453125 weighting, preserving total ensemble weight.
mechanism: Correctness-boundary vertical-shift TTA bisection
evidence_used: Weight 1.3125 achieved 9,348 correct, while 1.314453125 lost one prediction but lowered cross-entropy to 0.187707558441; their midpoint is the most informative remaining probe of the correctness boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 64.36736466712318, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.187707568359375, "validation_score": 9348.420979046796}

RECENT RESULT
hypothesis: Vertical weights of 1.31396484375 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.
change: Test the midpoint between the 9,348-correct 1.3134765625 weighting and the 9,347-correct 1.314453125 weighting, preserving center weight and total ensemble weight.
mechanism: Correctness-boundary vertical-shift TTA bisection
evidence_used: Increasing the vertical weight from 1.3125 to 1.3134765625 retained 9,348 correct and reduced cross-entropy, while 1.314453125 lost one prediction; their midpoint is the most informative remaining correctness-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 57.10681999986991, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18770756301879882, "validation_score": 9347.42097904869}

RECENT RESULT
hypothesis: Vertical weights of 1.313720703125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.
change: Test the midpoint between the successful 1.3134765625 weighting and the 9,347-correct 1.31396484375 weighting while preserving total ensemble weight.
mechanism: Correctness-boundary vertical-shift TTA bisection
evidence_used: Weight 1.3134765625 retained 9,348 correct with improved cross-entropy, whereas 1.31396484375 lost one prediction despite another small cross-entropy reduction; their midpoint is the most informative remaining boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 58.337812166893855, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.18770756568908692, "validation_score": 9347.420979047742}

RECENT RESULT
hypothesis: Vertical weights of 1.3135986328125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.
change: Test the midpoint between the successful 1.3134765625 weighting and the 9,347-correct 1.313720703125 weighting while preserving center and total ensemble weight.
mechanism: Correctness-boundary vertical-shift TTA bisection
evidence_used: Weight 1.3134765625 achieved 9,348 correct, whereas 1.313720703125 lost one prediction but slightly lowered cross-entropy; their midpoint is the most informative remaining correctness-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 58.61742058303207, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.187707568359375, "validation_score": 9347.420979046796}

RECENT RESULT
hypothesis: A class-specific attention head that learns where to pool feature evidence will exceed 9,348 correct predictions, unlike indiscriminate global pooling, while retaining the strong flattened spatial head.
change: Replace one dense bottleneck unit with a zero-initialized class-conditioned attention/value pooling branch, add learned class-specific spatial priors, and restore the best-verified TTA weights.
mechanism: Class-conditioned spatial attention residual
evidence_used: TTA bisection saturated at 9,348 correct, while global pooling fell to 9,085, showing that spatial organization is load-bearing. The old assumption is that a single flattened bottleneck should compute every prediction; this patch preserves that path but adds content-dependent, class-specific spatial aggregation instead of discarding location indiscriminately.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248524, "training_seconds": 65.05217441683635, "validation_accuracy": 0.9332, "validation_correct": 9332, "validation_cross_entropy": 0.187750728225708, "validation_score": 9332.420963749479}

RECENT RESULT
hypothesis: Matching training-crop frequencies to the best-verified inference weights will exceed 9,348 correct predictions by emphasizing the more reliable vertical translations without changing the model or evaluation ensemble.
change: Replace symmetric 3:1:1:1:1 translation sampling with a 48:21:21:11:11 categorical distribution, the nearest integer representation of the current 3:1.3134765625:1.3134765625:0.6865234375:0.6865234375 TTA weights.
mechanism: TTA-aligned anisotropic translation training
evidence_used: Vertical-favoring TTA improved correctness from 9,344 with symmetric weights to 9,348, while adaptive pooling and attention changes regressed; this directly motivates aligning the training exposure distribution with the successful anisotropy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 53.80732287513092, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.18945341796875, "validation_score": 9327.42036114441}

RECENT RESULT
hypothesis: Vertical weights of 1.31353759765625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.
change: Test the midpoint between the best verified 1.3134765625 weighting and the 9,347-correct 1.3135986328125 weighting, preserving center weight and total ensemble weight.
mechanism: Correctness-boundary vertical-shift TTA bisection
evidence_used: Weight 1.3134765625 achieved 9,348 correct with cross-entropy 0.187707568359375, whereas the nearest tested higher weight, 1.3135986328125, lost one prediction; their midpoint is the most informative remaining correctness-boundary probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 46.13268766691908, "validation_accuracy": 0.9347, "validation_correct": 9347, "validation_cross_entropy": 0.187707568359375, "validation_score": 9347.420979046796}



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
