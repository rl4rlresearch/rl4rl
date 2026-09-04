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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 73.77810429083183, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.22463382492065428, "validation_score": 9319.408285309311}
prior_hypothesis: Lowering classifier dropout from 0.10 to 0.05 will exceed 9,316 correct predictions by improving short-horizon feature utilization while retaining the beneficial 0.04 label smoothing.

## Recent verification evidence

RECENT RESULT
hypothesis: Widening the classifier hidden layer from 40 to 44 units will exceed 9,290 correct predictions by increasing layout-sensitive capacity while remaining below the 250,000-parameter ceiling.
change: Expand the successful flattening classifier to 44 hidden units, yielding 249,934 learned parameters while leaving training and regularization unchanged.
mechanism: Layout-preserving dense-head expansion
evidence_used: Reallocating dense-head parameters into an additional residual block reduced correctness, whereas the 40-unit layout-preserving head produced the strongest 9,290 result; this tests the opposite allocation direction using the remaining parameter budget.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 60.37565691699274, "validation_accuracy": 0.9291, "validation_correct": 9291, "validation_cross_entropy": 0.20789716110229492, "validation_score": 9291.413942524332}

RECENT RESULT
hypothesis: Increasing label smoothing from 0.02 to 0.04 will exceed 9,291 correct predictions by extending the demonstrated accuracy benefit of soft targets while retaining the successful 249,934-parameter architecture.
change: Increase cross-entropy label smoothing from 0.02 to 0.04; leave the model, batch size, optimizer, and schedule unchanged.
mechanism: Moderate target-distribution regularization
evidence_used: Removing 0.02 label smoothing reduced correctness from 9,290 to 9,270 despite improving cross-entropy, showing that soft-target regularization materially benefits the primary objective; this tests whether a moderately stronger value improves it further.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 69.85783979203552, "validation_accuracy": 0.9316, "validation_correct": 9316, "validation_cross_entropy": 0.22164978942871094, "validation_score": 9316.409282598275}

RECENT RESULT
hypothesis: Increasing label smoothing from 0.04 to 0.06 will exceed 9,316 correct predictions by continuing the verified gain from moderately stronger soft targets without changing the successful architecture or optimization.
change: Increase cross-entropy label smoothing from 0.04 to 0.06.
mechanism: Stronger soft-target regularization
evidence_used: On the same 249,934-parameter model, increasing label smoothing from 0.02 to 0.04 improved validation correctness from 9,291 to 9,316, while removing smoothing previously reduced correctness.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 67.5332382908091, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.24150046310424805, "validation_score": 9279.402738472405}

RECENT RESULT
hypothesis: Setting label smoothing to 0.038 will exceed 9,316 correct predictions by targeting the empirical peak implied by results at 0.02, 0.04, and 0.06.
change: Reduce cross-entropy label smoothing from 0.04 to 0.038 while preserving the strongest architecture and training configuration.
mechanism: Local label-smoothing optimum refinement
evidence_used: Correct predictions rose from 9,291 at 0.02 smoothing to 9,316 at 0.04, then fell to 9,279 at 0.06; quadratic interpolation places the local optimum near 0.038.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.09296700009145, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.22173592376708984, "validation_score": 9309.409253743197}

RECENT RESULT
hypothesis: Lowering classifier dropout from 0.10 to 0.05 will exceed 9,316 correct predictions by improving short-horizon feature utilization while retaining the beneficial 0.04 label smoothing.
change: Reduce the dense classifier’s dropout probability to 0.05; preserve the architecture, optimizer, schedule, and label smoothing.
mechanism: Reduced stochastic head regularization
evidence_used: Label smoothing improved correctness at 0.04 but regressed at 0.06, indicating excessive regularization is harmful; tuning the still-untested dropout strength isolates another source of regularization without disturbing the strongest configuration.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 73.77810429083183, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.22463382492065428, "validation_score": 9319.408285309311}

RECENT RESULT
hypothesis: Eliminating classifier dropout will exceed 9,319 correct predictions by improving short-horizon feature utilization while retaining the accuracy-enhancing 0.04 label smoothing.
change: Set the dense classifier’s dropout probability from 0.05 to 0.0 while preserving all other architecture and training settings.
mechanism: Remove stochastic head regularization
evidence_used: Reducing dropout from 0.10 to 0.05 improved correctness from 9,316 to 9,319, while stronger label smoothing regressed, indicating that less regularization is beneficial near the current configuration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 68.40263845818117, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.22428358192443848, "validation_score": 9276.408402111554}

RECENT RESULT
hypothesis: Setting classifier dropout to 0.07 will exceed 9,319 correct predictions by balancing the under-regularization observed at 0.0 with the slight over-regularization observed at 0.10.
change: Increase the dense classifier’s dropout probability from 0.05 to 0.07 while preserving the strongest architecture, optimizer, schedule, and label smoothing.
mechanism: Local classifier-dropout optimum refinement
evidence_used: Dropout 0.05 achieved 9,319 correct versus 9,276 at 0.0 and 9,316 at 0.10; quadratic interpolation of these three results places the estimated local optimum near 0.072.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 57.4860969169531, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.22331139030456543, "validation_score": 9309.408726677413}

RECENT RESULT
hypothesis: Setting classifier dropout to 0.045 will exceed 9,319 correct predictions by moving toward the local optimum implied by the results at 0.0, 0.05, and 0.07.
change: Reduce the dense classifier’s dropout probability from 0.05 to 0.045 while preserving all other settings.
mechanism: Fine-grained classifier-dropout refinement
evidence_used: Dropout 0.05 achieved 9,319 correct, outperforming both 0.0 at 9,276 and 0.07 at 9,309; interpolation of those nearby results places the estimated peak slightly below 0.05.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 53.25774291693233, "validation_accuracy": 0.9305, "validation_correct": 9305, "validation_cross_entropy": 0.2229373416900635, "validation_score": 9305.408851690889}

RECENT RESULT
hypothesis: Initializing AdamW at the schedule’s 20% warmup rate will exceed 9,319 correct predictions by preventing the first update from occurring at full peak learning rate before abruptly dropping.
change: Change AdamW’s initial learning rate from 2.0e-3 to 4.0e-4; retain the existing warmup, cosine schedule, architecture, and regularization.
mechanism: Warmup-consistent optimizer initialization
evidence_used: The strongest configuration achieved 9,319 correct, while subsequent fine-grained dropout changes regressed. Its schedule specifies a 0.2× starting multiplier, but the optimizer currently performs its first update at the full 2.0e-3 peak, making warmup consistency an untested optimization improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 67.19492674991488, "validation_accuracy": 0.9298, "validation_correct": 9298, "validation_cross_entropy": 0.22133691635131836, "validation_score": 9298.40938744527}

RECENT RESULT
hypothesis: Removing the 5% learning-rate warmup will exceed 9,319 correct predictions by preserving aggressive early optimization throughout the short two-pass training horizon.
change: Replace the warmup-plus-cosine schedule with cosine decay from the full 2.0e-3 learning rate to 3% of peak.
mechanism: Warmup-free cosine decay
evidence_used: Making the first update warmup-consistent reduced correctness from 9,319 to 9,298; this suggests the full-rate initial update is beneficial and motivates testing full-rate early training rather than a 5% low-rate ramp.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 68.63605229184031, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.22911708641052247, "validation_score": 9275.406796069738}

RECENT RESULT
hypothesis: Retaining 0.04 smoothing for 90% of training and annealing it to zero over the final 10% will exceed 9,319 correct predictions by preserving its verified generalization benefit while sharpening class margins near convergence.
change: Replace constant label smoothing with a late linear decay from 0.04 to 0.0.
mechanism: Terminal label-smoothing annealing
evidence_used: Constant 0.04 smoothing achieved 9,316–9,319 correct, while hard labels produced substantially lower cross-entropy but only 9,270 correct; a terminal anneal tests whether their complementary benefits can be combined.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 74.91917475010268, "validation_accuracy": 0.9309, "validation_correct": 9309, "validation_cross_entropy": 0.2045325019836426, "validation_score": 9309.415098803209}

RECENT RESULT
hypothesis: Ramping label smoothing from 0 to 0.04 during the first 10% of training will exceed 9,319 correct predictions by strengthening early class-learning gradients while retaining smoothing throughout convergence.
change: Linearly warm label smoothing from 0 to 0.04 over the first 10% of optimizer steps, then hold it at 0.04.
mechanism: Early hard-target smoothing ramp
evidence_used: Constant 0.04 smoothing reached 9,319 correct, while annealing it to zero late fell to 9,309; this indicates smoothing is especially valuable near convergence, motivating the reverse schedule.
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
