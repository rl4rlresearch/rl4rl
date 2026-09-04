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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 69.16518004215322, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877068977355957, "validation_score": 9348.420979284496}
prior_hypothesis: Redistributing 1/1024 ensemble weight from the downward-shifted crop to the upward-shifted crop will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 63.23315116693266, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877075828552246, "validation_score": 9348.420979041657}
prior_hypothesis: Increasing the vertical-shift preference slightly beyond the best 1.25/0.75 weighting will exceed 9,346 correct predictions or retain 9,346 while lowering cross-entropy below 0.18771290.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.60296387504786, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877066421508789, "validation_score": 9348.420979375087}
prior_hypothesis: An additional 1/8192 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770672912597655.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 51.1557595001068, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770757064819335, "validation_score": 9348.420979045984}
prior_hypothesis: Vertical weights of 1.3134918212890625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Vertical weights of 1.313507080078125 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.
change: Restore the favorable TTA region and test the midpoint between the best verified 1.3134765625 weighting and the 9,347-correct 1.31353759765625 weighting, preserving center weight and total ensemble weight.
mechanism: Correctness-boundary vertical-shift TTA bisection
evidence_used: Weight 1.3134765625 achieved 9,348 correct, while the nearest tested higher weight, 1.31353759765625, lost one prediction; their midpoint is the most informative remaining correctness-boundary probe.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Vertical weights of 1.3134918212890625 will retain 9,348 correct predictions while lowering validation cross-entropy below 0.187707568359375.
change: Restore the best-performing TTA neighborhood and test a conservative quarter-step above the verified 1.3134765625 weight, preserving center and total ensemble weight.
mechanism: Conservative correctness-boundary TTA refinement
evidence_used: Weight 1.3134765625 achieved 9,348 correct, while 1.31353759765625 lost one prediction; the intervening midpoint timed out without producing evidence, so this lower-side probe offers the best chance of a correctness-preserving cross-entropy gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 51.1557595001068, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770757064819335, "validation_score": 9348.420979045984}

RECENT RESULT
hypothesis: Redistributing 1/1024 ensemble weight from the downward-shifted crop to the upward-shifted crop will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy.
change: Break the unexplored symmetry between vertical crops while preserving their combined weight, center weight, horizontal weights, and temperature.
mechanism: Directional vertical-shift TTA reweighting
evidence_used: Favoring vertical over horizontal translations improved correctness from 9,344 to 9,348, but symmetric vertical-weight refinement reached a correctness boundary; directional vertical weighting is the nearest untested TTA axis.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 69.16518004215322, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877068977355957, "validation_score": 9348.420979284496}

RECENT RESULT
hypothesis: Moving another 1/2048 ensemble weight from the downward-shifted crop to the upward-shifted crop will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877068977355957.
change: Restore the best verified ensemble mean and horizontal weights, then increase the successful vertical directional asymmetry by half the previously tested step while preserving total ensemble weight.
mechanism: Conservative directional vertical-shift TTA extrapolation
evidence_used: A 1/1024 directional redistribution retained 9,348 correct and improved cross-entropy from approximately 0.187707568 to 0.187706898; a conservative half-step extrapolation directly tests whether that gain continues before risking a larger decision-boundary crossing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The verified asymmetric crop weights will retain 9,348 correct predictions while lowering cross-entropy below the current 0.187707582855.
change: Restore the best-verified ensemble weights by transferring 1/1024 weight between vertical crops and preserving total weight.
mechanism: Directional vertical-shift TTA reweighting
evidence_used: Reference Design 1 achieved 9,348 correct with cross-entropy 0.187706897736, the highest available validation_score.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Extending the successful vertical asymmetry by 1/4096 will retain 9,348 correct predictions while lowering cross-entropy below 0.1877068977355957.
change: Restore the best verified total vertical and horizontal weights, then transfer an additional 1/4096 weight between the two vertical crops while preserving total ensemble weight.
mechanism: Conservative directional vertical-shift TTA extrapolation
evidence_used: Reference Design 2’s 1/1024 directional transfer retained 9,348 correct and achieved the best verified cross-entropy; a larger additional 1/2048 transfer timed out, so a smaller extrapolation is the most informative low-risk probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 57.25442320900038, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.18770672912597655, "validation_score": 9348.420979344259}

RECENT RESULT
hypothesis: An additional 1/8192 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770672912597655.
change: Transfer 1/8192 ensemble weight from the first vertical crop to the second, preserving center, horizontal, and total weights.
mechanism: Conservative directional vertical-shift TTA extrapolation
evidence_used: The current additional 1/4096 transfer retained 9,348 correct and improved cross-entropy from 0.1877068977355957 to 0.18770672912597655; halving the next step conservatively tests whether this directional gain continues.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249881, "training_seconds": 59.60296387504786, "validation_accuracy": 0.9348, "validation_correct": 9348, "validation_cross_entropy": 0.1877066421508789, "validation_score": 9348.420979375087}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
