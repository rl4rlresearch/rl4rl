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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 77.28444195794873, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2356298973083496, "validation_score": 9252.40465191162}
prior_hypothesis: Averaging snapshots from only the final 5% of training will exceed 9,252 correct predictions by retaining the variance reduction of successful weight averaging while reducing bias from earlier parameters and mismatch with final BatchNorm statistics.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing centered-crop probability from 5/13 to 3/7 while retaining only cardinal translations will exceed 9,249 correct predictions by continuing the demonstrated benefit of modestly reducing translation magnitude without reaching the previously harmful 4/9 center concentration.
change: Change cross-shaped crop weights from 5:2 center-to-cardinal to 3:1, giving probability 3/7 to the centered crop and 1/7 to each cardinal shift.
mechanism: Moderately stronger center-biased cross translation augmentation
evidence_used: Increasing the cardinal-only center weight from 2:1 to 5:2 improved validation_correct from 9,232 to 9,238, while removing diagonal inference views subsequently raised it to 9,249; 3:1 is the next controlled center-weight increase and remains below the unsuccessful 4/9 centered probability.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing centered-crop probability from 5/13 to 2/5 will exceed 9,249 correct predictions by continuing the observed benefit of modest center bias while remaining more conservative than the unverified 3/7 setting.
change: Change cardinal-only crop weights from 5:2 to 8:3, assigning probability 2/5 to the centered crop and 3/20 to each cardinal translation.
mechanism: Intermediate center-biased cross translation augmentation
evidence_used: Increasing center probability within cross-shaped augmentation from 1/3 to 5/13 improved validation_correct from 9,232 to 9,238; the next stronger 3/7 setting timed out, motivating this intermediate refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising centered-crop probability from 5/13 to 2/5 will exceed 9,249 correct predictions, while reusing centered inference logits will help the previously timed-out setting complete.
change: Change cardinal-only crop weights from 5:2 to 8:3 and eliminate the redundant second forward pass for the centered unflipped validation view.
mechanism: Runtime-efficient intermediate center-biased cross augmentation
evidence_used: Increasing center probability from 1/3 to 5/13 improved validation_correct from 9,232 to 9,238; the 2/5 experiment timed out without accuracy evidence, and the current evaluation unnecessarily computes centered logits twice.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging logits across the same cardinal and flipped views will exceed 9,249 correct predictions by rewarding cross-view class agreement instead of allowing one high-probability view to dominate the arithmetic probability ensemble.
change: Replace probability-space averaging with logit-space averaging while reusing the already-computed centered logits, preserving all ten inference views and the training procedure.
mechanism: Geometric-mean cardinal-view ensemble
evidence_used: Cardinal-only inference improved validation_correct from 9,238 to 9,249, confirming these views contain useful complementary evidence; spatial probability reweighting reduced accuracy, so changing only the aggregation rule is the most targeted remaining inference test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging logits across the ten cardinal and horizontally flipped views will exceed 9,249 correct predictions, while single-pass vectorization will allow the previously timed-out aggregation test to complete.
change: Replace sequential probability averaging and the redundant centered forward pass with one batched forward pass followed by logit averaging.
mechanism: Vectorized geometric-mean cardinal-view ensemble
evidence_used: Cardinal-only probability averaging produced the best result of 9,249 correct; the subsequent logit-averaging test timed out, while prior evidence showed that an even larger vectorized 26-view ensemble completed successfully.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reallocating dense-head parameters to a 7×7 convolutional refinement layer will exceed 9,249 correct predictions by learning stronger spatial features while preserving the successful training and inference procedure.
change: Add a third 64-channel convolution after the second pooling stage and reduce the classifier bottleneck from 48 to 46 units, keeping the model under the 250,000-parameter ceiling.
mechanism: Late convolutional refinement with dense-head reallocation
evidence_used: The best verified design reached 9,249 correct with 216,346 parameters, while subsequent inference and crop-weight refinements timed out or failed to improve; the current 150,576-parameter first dense layer leaves convolutional feature learning comparatively underallocated.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Widening the classifier bottleneck from 48 to 56 units will exceed 9,249 correct predictions by using the remaining parameter budget to preserve more learned spatial features without the runtime cost of another convolution.
change: Increase both classifier-layer dimensions and batch-normalization width from 48 to 56, yielding approximately 241,538 learned parameters.
mechanism: Compute-efficient dense bottleneck expansion
evidence_used: The best design uses only 216,346 of 250,000 allowed parameters, while the attempted convolutional expansion timed out; widening the existing dense bottleneck adds capacity with negligible extra spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging evenly spaced parameter snapshots from the final 10% of the cosine-decayed trajectory will exceed 9,249 correct predictions by reducing terminal mini-batch noise without adding model parameters or inference work.
change: Accumulate every fourth low-learning-rate checkpoint during the final 10% of training and replace the final parameters with their arithmetic average.
mechanism: Sparse late-checkpoint weight averaging
evidence_used: The best design reached 9,249 correct, while added capacity and altered inference aggregation repeatedly timed out; this isolates checkpoint stability with negligible computation and preserves the successful architecture, augmentation, and ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 70.05571620794944, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2360583267211914, "validation_score": 9252.40451165547}

RECENT RESULT
hypothesis: Averaging BatchNorm running statistics alongside the successful late parameter snapshots will exceed 9,252 correct predictions by evaluating the averaged weights with activation statistics from the same trajectory window.
change: Retain the model on the optimizer and jointly average/copy BatchNorm running means and variances at every existing late checkpoint.
mechanism: BatchNorm-aligned late-checkpoint averaging
evidence_used: Sparse late-checkpoint parameter averaging improved validation_correct from 9,249 to 9,252, but the current implementation leaves BatchNorm statistics at their final, unaveraged values; aligning those statistics directly refines the only recently successful mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 77.08310595783405, "validation_accuracy": 0.9248, "validation_correct": 9248, "validation_cross_entropy": 0.2346428108215332, "validation_score": 9248.4049754274}

RECENT RESULT
hypothesis: Averaging snapshots from only the final 5% of training will exceed 9,252 correct predictions by retaining the variance reduction of successful weight averaging while reducing bias from earlier parameters and mismatch with final BatchNorm statistics.
change: Move the start of sparse parameter averaging from 90% to 95% of the trajectory, preserving its cadence and all other training and inference behavior.
mechanism: Narrow late-checkpoint weight averaging
evidence_used: Final-10% parameter averaging improved validation_correct from 9,249 to 9,252, whereas also averaging BatchNorm statistics reduced it to 9,248; narrowing the parameter window is a targeted way to align averaged weights more closely with the beneficial final BatchNorm state.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 77.28444195794873, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2356298973083496, "validation_score": 9252.40465191162}

RECENT RESULT
hypothesis: Averaging snapshots from only the final 2.5% of training will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by further reducing mismatch between averaged parameters and final BatchNorm statistics.
change: Move the start of sparse parameter averaging from 95% to 97.5% of the trajectory while preserving its cadence and all other behavior.
mechanism: Tighter terminal-checkpoint weight averaging
evidence_used: Narrowing the averaging window from the final 10% to 5% retained 9,252 correct predictions and lowered cross-entropy from 0.2360583 to 0.2356299, while averaging BatchNorm statistics hurt accuracy; this motivates another controlled narrowing of parameter averaging alone.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging equally many snapshots over only the final 2.5% of training will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by reducing BatchNorm mismatch without sacrificing the variance reduction of the successful final-5% average.
change: Start parameter averaging at 97.5% progress and double its sampling frequency from every four steps to every two steps, preserving approximately the current number of averaged snapshots.
mechanism: Dense terminal-checkpoint weight averaging
evidence_used: Narrowing averaging from the final 10% to 5% retained 9,252 correct and reduced cross-entropy from 0.2360583 to 0.2356299; the prior 2.5% attempt retained the four-step cadence and timed out, so preserving snapshot count is the most targeted next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.79407591698691, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.23564663162231445, "validation_score": 9247.404646431434}



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
