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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 79.18238341598772, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.21136927757263183, "validation_score": 9282.412756051566}
prior_hypothesis: Training the same two-view logit ensemble used at validation will exceed 9,237 correct predictions by directly optimizing complementary original and mirrored predictions.

## Recent verification evidence

RECENT RESULT
hypothesis: A 216,250-parameter residual CNN trained with batch size 128, horizontal-flip augmentation, and a high-floor cosine schedule will exceed the baseline’s 8,928 correct predictions.
change: Replace the shallow CNN with two residual convolutional stages, halve batch size, add per-image horizontal flips, reduce label smoothing, and use a longer high-learning-rate phase.
mechanism: Wider batch-normalized residual CNN with more frequent updates
evidence_used: The current 105,866-parameter model reaches 8,928/10,000 after only 392 optimizer steps, motivating greater feature capacity and twice as many updates within the same 100,000-example budget.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216250, "training_seconds": 32.89338720915839, "validation_accuracy": 0.9194, "validation_correct": 9194, "validation_cross_entropy": 0.23204684257507324, "validation_score": 9194.4058287256}

RECENT RESULT
hypothesis: Averaging predictions for each validation image and its horizontal mirror will exceed 9,194 correct predictions because training already exposes the model to both orientations.
change: Refactor the single-view forward pass and ensemble original and flipped views only during evaluation, without changing training or parameter count.
mechanism: Evaluation-time horizontal-flip logit ensemble
evidence_used: The current flip-augmented residual CNN achieves 9,194/10,000; its learned flip invariance makes a matched two-view ensemble a low-risk way to reduce prediction variance.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216250, "training_seconds": 29.31515641696751, "validation_accuracy": 0.9212, "validation_correct": 9212, "validation_cross_entropy": 0.22716464080810547, "validation_score": 9212.407443291122}

RECENT RESULT
hypothesis: Deepening both residual branches while retaining the proven flip ensemble will exceed 9,212 correct validation predictions.
change: Add a second convolution to each residual block and narrow the hidden classifier from 48 to 44 units, yielding 249,934 learned parameters.
mechanism: Parameter-budgeted two-convolution residual blocks
evidence_used: The wider residual CNN improved validation correct from 8,928 to 9,194, and flip ensembling raised it to 9,212; this motivates investing the remaining parameter budget in additional residual feature extraction while preserving the successful evaluation method.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249934, "training_seconds": 36.005177167011425, "validation_accuracy": 0.9196, "validation_correct": 9196, "validation_cross_entropy": 0.2364686004638672, "validation_score": 9196.404377434099}

RECENT RESULT
hypothesis: Averaging center and four one-pixel translations, each with its horizontal mirror, will exceed 9,212 correct predictions by reducing the max-pooling phase sensitivity that remains after the successful mirror ensemble.
change: Expand evaluation-only augmentation from two mirror views to ten views spanning the original position and one-pixel shifts in four directions; leave training and parameter count unchanged.
mechanism: Pooling-phase translation ensemble
evidence_used: Evaluation-time mirror averaging improved the same model from 9,194 to 9,212 correct without retraining, showing that view ensembling reduces prediction variance; small translations target an additional invariance relevant to the model’s two max-pooling operations.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216250, "training_seconds": 30.910406542010605, "validation_accuracy": 0.9187, "validation_correct": 9187, "validation_cross_entropy": 0.24127032165527343, "validation_score": 9187.402813143339}

RECENT RESULT
hypothesis: Increasing optimizer steps from 782 to about 1,563 while scaling the learning rate to 2.1e-3 will exceed 9,212 correct validation predictions.
change: Halve batch size to 64 and reduce the AdamW base and scheduled learning rates for stable, more frequent updates.
mechanism: Small-batch optimization with square-root learning-rate scaling
evidence_used: Expanding the model to 249,934 parameters regressed to 9,196 correct, while the successful 216,250-parameter design still receives only 782 updates; this motivates spending computation on optimization rather than additional capacity.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 60.97917112498544, "validation_accuracy": 0.9215, "validation_correct": 9215, "validation_cross_entropy": 0.2251504177093506, "validation_score": 9215.408113153106}

RECENT RESULT
hypothesis: Averaging the final half of the proven small-batch optimization trajectory will exceed 9,215 correct predictions by reducing parameter and BatchNorm-state variance without adding learned parameters.
change: Track an EMA of the complete model state after 50% of training and install the averaged state after the final optimizer step.
mechanism: Late-training exponential weight averaging
evidence_used: Halving the batch size improved the 216,250-parameter model from 9,212 to 9,215 correct while the larger model regressed, indicating that further gains are more likely from stabilizing optimization than increasing capacity.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 55.16018633288331, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.2225617126464844, "validation_score": 9237.408977309553}

RECENT RESULT
hypothesis: Increasing the EMA horizon from roughly 100 to 200 optimizer steps will exceed 9,237 correct predictions by further suppressing small-batch parameter and BatchNorm-state variance while retaining emphasis on the converged trajectory.
change: Reduce the EMA interpolation rate from 0.01 to 0.005, preserving the proven architecture, schedule, augmentation, and averaging start point.
mechanism: Longer-horizon late-training exponential weight averaging
evidence_used: Late-training EMA improved the same small-batch model from 9,215 to 9,237 correct; extending that successful averaging mechanism is the most direct test of whether additional trajectory smoothing yields further gains.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 54.43861262500286, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.2232889549255371, "validation_score": 9233.408734173547}

RECENT RESULT
hypothesis: Training the same two-view logit ensemble used at validation will exceed 9,237 correct predictions by directly optimizing complementary original and mirrored predictions.
change: Evaluate every training image in both orientations, average the paired logits, and compute cross-entropy on the resulting ensemble prediction.
mechanism: Paired-view ensemble training
evidence_used: Evaluation-time mirror averaging improved the same architecture from 9,194 to 9,212 correct; directly training that successful predictor is an informative way to align optimization with validation behavior while preserving the proven EMA configuration.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 79.18238341598772, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.21136927757263183, "validation_score": 9282.412756051566}



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
