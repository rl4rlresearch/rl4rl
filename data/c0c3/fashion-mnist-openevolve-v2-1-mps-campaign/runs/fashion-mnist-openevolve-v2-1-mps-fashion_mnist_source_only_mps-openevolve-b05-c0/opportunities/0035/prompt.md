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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 82.23966912506148, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.24194595336914063, "validation_score": 9254.402594008736}
prior_hypothesis: Updating the tail EMA once every four batch-32 steps will exceed 9,250 correct predictions by matching the successful batch-128 EMA’s examples-based averaging horizon while reducing averaging overhead.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging logits from each validation image and its horizontal reflection will exceed 9,254 correct predictions by reducing orientation-sensitive errors without altering the proven training trajectory.
change: Preserve training behavior and add a two-view horizontal-flip ensemble only while the model is in evaluation mode.
mechanism: Horizontal-reflection test-time ensembling
evidence_used: The strided-EMA batch-32 design achieved 9,254 correct, while several attempts to alter training timed out; this validation-only change retains that successful procedure and adds negligible training cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 67.26239279098809, "validation_accuracy": 0.8873, "validation_correct": 8873, "validation_cross_entropy": 0.40361724166870117, "validation_score": 8873.356222469458}

RECENT RESULT
hypothesis: Applying 10% dropout to the classifier hidden layer will exceed 9,254 correct predictions by reducing co-adaptation in the parameter-dominant fully connected head.
change: Insert dropout after the classifier’s hidden GELU activation while preserving all other architecture and training settings.
mechanism: Mild hidden-representation dropout
evidence_used: Label smoothing 0.05 outperformed both hard targets and 0.10 smoothing, showing that moderate regularization helps; hidden dropout tests an orthogonal, targeted form of moderate regularization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the classifier hidden width from 88 to 92 will exceed 9,254 correct predictions by reducing compression of the 2,352-dimensional feature representation while remaining below the parameter ceiling.
change: Widen the classifier’s hidden layer by four units, increasing learned parameters from 239,634 to 249,086 without changing training runtime-sensitive settings.
mechanism: Expanded classifier bottleneck
evidence_used: The best batch-32 design leaves 10,366 parameters unused, while several changes adding training work timed out; allocating nearly all remaining capacity to the parameter-dominant classifier is a low-compute test of whether its 88-unit bottleneck limits accuracy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reallocating parameters from the oversized dense head into wider convolutional features will exceed 9,254 correct predictions by learning richer spatial representations while reducing per-image multiply-accumulates.
change: Widen feature channels from 24/48/48 to 32/64/64, move the second pooling operation before the final convolution, and narrow the classifier hidden layer from 88 to 60; this uses 244,894 learned parameters.
mechanism: Compute-neutral feature-capacity reallocation
evidence_used: The best design assigns 207,954 of 239,634 parameters to its dense head, while runtime-increasing experiments repeatedly timed out. This redistribution increases convolutional capacity by roughly 77% while reducing approximate convolution-plus-linear compute from 6.47M to 5.83M operations per image.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing label smoothing from 0.05 to 0.04 will exceed 9,254 correct predictions by retaining beneficial regularization while reducing target underconfidence.
change: Change only the cross-entropy label-smoothing coefficient from 0.05 to 0.04.
mechanism: Fine-grained target regularization
evidence_used: Label smoothing 0.05 outperformed both hard targets and 0.10, locating the useful region near mild smoothing; a small downward refinement is runtime-neutral and may improve both classification decisions and tie-breaking cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing AdamW’s second-moment decay from 0.999 to 0.99 will exceed 9,254 correct predictions by adapting gradient scaling more quickly within the fixed 3,126-step training budget.
change: Set AdamW betas explicitly to `(0.9, 0.99)` while preserving the architecture, learning-rate schedule, loss, batch size, and strided EMA.
mechanism: Faster second-moment adaptation
evidence_used: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that finite-budget optimization benefits from more responsive updates; a shorter second-moment horizon tests that mechanism without adding computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing batch size from 32 to 31 will exceed 9,254 correct predictions by providing about 100 additional optimizer updates while remaining close to the proven runtime and EMA exposure horizon.
change: Change only the training batch size from 32 to 31.
mechanism: Slightly higher-update small-batch optimization
evidence_used: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250; this minimal reduction tests the same mechanism with substantially less runtime risk than the timed-out batch-24 and batch-28 designs.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting BatchNorm momentum to 0.01 will exceed 9,254 correct predictions by reducing batch-to-batch noise in evaluation statistics while remaining responsive to the late training trajectory represented by the parameter EMA.
change: Change all three BatchNorm layers from the default 0.1 momentum to 0.01 without altering model parameters, training compute, or the proven optimizer and EMA procedure.
mechanism: Moderately smoothed BatchNorm evaluation statistics
evidence_used: BatchNorm previously increased correct predictions from 9,170 to 9,210, and the current parameter EMA raised the batch-32 result from 9,250 to 9,254; this motivates stabilizing the running statistics used with those averaged parameters, while choosing a more responsive horizon than the unverified 0.0025 attempt.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Keeping 0.05 label smoothing for the first 75% of training and annealing it to zero will exceed 9,254 correct predictions by preserving proven early regularization while allowing harder late target fitting.
change: Replace constant label smoothing with a cosine decay from 0.05 to 0 during the final quarter of optimizer steps.
mechanism: Late-phase label-smoothing annealing
evidence_used: Label smoothing 0.05 outperformed both hard targets and 0.10, while smaller batches showed that additional finite-budget optimization improves accuracy; a late anneal conservatively combines the proven regularization with stronger final fitting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying deterministic one-pixel translations during the first half of training will exceed 9,254 correct predictions by adding mild spatial regularization, while the clean second half preserves distribution-matched fitting and EMA averaging.
change: Translate each early training batch in one of eight directions using replicated borders; leave the second half of training unchanged.
mechanism: Early one-pixel translation augmentation with clean-tail fitting
evidence_used: Moderate label smoothing outperformed both hard targets and 0.10 smoothing, indicating that mild regularization helps, while horizontal-reflection ensembling fell to 8,873 correct; small translations test a less destructive spatial invariance without altering the proven batch size, optimizer, or EMA tail.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics alongside the proven tail parameter EMA will exceed 9,254 correct predictions by eliminating the mismatch between averaged weights and final-step normalization statistics.
change: Retain the model on the optimizer and apply the existing strided EMA to floating-point model buffers, restoring both averaged parameters and BatchNorm statistics before validation.
mechanism: BatchNorm-consistent parameter EMA
evidence_used: BatchNorm previously improved correct predictions from 9,170 to 9,210, and the parameter EMA improved the batch-32 result from 9,250 to 9,254; these results motivate making the two beneficial mechanisms internally consistent.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restricting tail EMA to the parameter-dominant classifier will exceed 9,254 correct predictions by retaining weight-averaging benefits while keeping convolution and BatchNorm parameters aligned with final running statistics.
change: Store the classifier parameters as the EMA targets and leave the entire feature extractor at its final trained state.
mechanism: Classifier-only tail EMA
evidence_used: Full-parameter EMA improved the batch-32 result from 9,250 to 9,254, while BatchNorm previously added 40 correct predictions and the classifier contains 207,954 of 239,634 parameters; this motivates concentrating EMA on the dense head without averaging parameters that determine BatchNorm feature statistics.
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
