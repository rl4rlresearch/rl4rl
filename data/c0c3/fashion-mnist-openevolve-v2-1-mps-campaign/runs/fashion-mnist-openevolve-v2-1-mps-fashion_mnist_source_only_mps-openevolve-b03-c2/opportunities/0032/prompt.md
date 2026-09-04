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
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247538, "training_seconds": 22.307899000123143, "validation_accuracy": 0.9105, "validation_correct": 9105, "validation_cross_entropy": 0.25879054794311523, "validation_score": 9105.39720666859}
prior_hypothesis: Expanding convolutional capacity to 247,538 parameters, doubling optimizer updates via batch size 128, and using one augmented pass followed by clean fine-tuning will exceed 8,928 correct validation predictions.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249550, "training_seconds": 63.629130667075515, "validation_accuracy": 0.9119, "validation_correct": 9119, "validation_cross_entropy": 0.29910097885131837, "validation_score": 9119.384881551272}
prior_hypothesis: Ten-view inference over centered and cardinal shifts with horizontal mirrors will exceed 9,112 correct predictions while preserving ample runtime headroom.

REFERENCE DESIGN 3
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 81.30000304197893, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.24333785018920898, "validation_score": 9214.402143311188}
prior_hypothesis: Replacing ±2-pixel and diagonal training crops with the exact five spatial views used by ten-view inference will exceed 9,208 correct predictions without increasing training time.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing residual width from 112 to 108 channels while reallocating parameters to a 76-unit classifier will give the proven 0.995 EMA enough runtime headroom to finish and exceed 9,214 correct predictions.
change: Adopt Reference Design 3’s batch-32 matched-view, flip-paired training and ten-view inference; narrow its expensive convolutions, widen its classifier, batch mirrored inference, and install a fused per-step EMA for final evaluation.
mechanism: Capacity-reallocated narrow residual network funding exact parameter EMA
evidence_used: Reference Design 3 finished with 9,214 correct, while exact 0.995 EMA reached 9,241 correct but timed out at 82.89 seconds; modestly narrowing the dominant convolutions targets that runtime failure while retaining 249,090 learned parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Processing all ten validation views in one forward call will provide enough runtime headroom for the previously observed 9,241-correct 0.995 EMA to finish verification and exceed 9,214 correct predictions.
change: Preserve the proven training rule, maintain a fused per-step EMA of learned parameters, install it after the final update, and concatenate all spatial and mirrored inference views into one batch.
mechanism: Fully batched ten-view inference funding exact parameter EMA
evidence_used: Exact EMA reached 9,241 correct versus the qualified 9,214 but timed out at 82.89 seconds; batching only each flip pair still timed out, motivating the stronger reduction from five inference calls to one.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Single-view random flips during the augmented half, exact flip pairing during clean fine-tuning, and a final-10% learning-rate cooldown will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt the proven residual ten-view design at batch 32, match its cardinal training views, halve duplicated forward work during the augmented half, retain late flip-paired supervision, and cool the learning rate to zero near completion.
mechanism: Compute-reallocated late flip pairing with terminal cooldown
evidence_used: Reference Design 3 achieved 9,214 correct, while EMA reached 9,241 but repeatedly timed out; random early flips remove 25% of its training forward work, and terminal cooling targets the same late-update noise without parameter-averaging overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Giving the center view twice the inference weight of each shifted view will exceed 9,214 correct predictions because the center view dominates the clean second half of training.
change: Adopt Reference Design 3’s batch-32 residual model, matched cardinal-view curriculum, flip-paired supervision, and cosine schedule; batch each inference flip pair and double the center-view probability weight.
mechanism: Curriculum-weighted flip-paired ten-view inference
evidence_used: Reference Design 3 achieved 9,214 correct, with matched spatial training improving 9,208 to 9,214; its second half trains exclusively on centered images, while equal-weight inference does not reflect that exposure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch size 40 will preserve most of Reference Design 3’s dense optimization while creating enough runtime headroom to finish, and cooling its final 10% of updates to zero will exceed 9,214 correct predictions by reducing the late-update noise implicated by EMA’s 9,241-correct result.
change: Adopt the near-250K residual model, matched cardinal-view curriculum, flip-paired supervision, and ten-view inference; increase batch size to 40, batch all inference views into one forward pass, and add a final-10% learning-rate cooldown.
mechanism: Throughput-funded terminal learning-rate annealing
evidence_used: Reference Design 3 achieved 9,214 correct, while 0.995 EMA achieved 9,241 but timed out at 82.89 seconds; batch 40 reduces optimizer steps from 3,126 to 2,500 without a partial batch, and terminal annealing targets the same late-weight instability without EMA updates.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering the terminal learning rate from 2.0e-4 to 4.0e-5 will reduce late-update noise and exceed 9,214 correct predictions without increasing runtime.
change: Reduce the cosine schedule’s learning-rate floor from 10% to 2% while retaining the existing peak rate and training procedure.
mechanism: Near-zero cosine learning-rate floor
evidence_used: Parameter EMA reached 9,241 correct versus the current 9,214, indicating that late-weight stabilization is beneficial; a lower terminal learning rate targets the same instability without EMA overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Ten-view inference over centered and cardinal shifts with horizontal mirrors will exceed 9,112 correct predictions while preserving ample runtime headroom.
change: Keep the verified 43-second training procedure unchanged and probability-average ten augmentation-matched views during evaluation, batching each mirrored pair.
mechanism: Training-aligned ten-view probability ensemble
evidence_used: The current model already trains on all ±2-pixel offsets and random flips, while Reference Design 3’s matched ten-view inference contributed to the strongest completed result of 9,214 correct.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249550, "training_seconds": 63.629130667075515, "validation_accuracy": 0.9119, "validation_correct": 9119, "validation_cross_entropy": 0.29910097885131837, "validation_score": 9119.384881551272}

RECENT RESULT
hypothesis: A 104-channel residual trunk with a widened 90-unit classifier, flip-paired matched-view training, and a 2%-floor cosine schedule will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt the strongest batch-32 residual training design, reduce its dominant convolution cost by roughly 16%, reallocate the saved parameters to the classifier, batch mirrored inference pairs, and lower the terminal learning rate to 4e-5.
mechanism: Compute-funded near-zero cosine stabilization
evidence_used: Reference Design 3 reached 9,214 correct, while EMA reached 9,241 but repeatedly timed out; the 249,948-parameter reallocation preserves near-ceiling capacity while directly reducing convolutional runtime and using the tested low-floor schedule to target late-update noise without EMA overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch-40 flip-paired training with example-scale-equivalent EMA sampled every four updates will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt the proven residual matched-view ten-view design, reduce optimizer steps to 2,500, slightly scale its learning rate, and install the strided equivalent of the accuracy-improving 0.995 batch-32 EMA.
mechanism: Throughput-funded strided parameter EMA
evidence_used: Reference Design 3 achieved 9,214 correct, while per-step 0.995 EMA reached 9,241 but timed out at 82.89 seconds; batch 40 reduces optimizer steps by 20%, and four-step EMA sampling preserves the same approximate example-space averaging horizon with one-quarter as many averaging operations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the expensive 112-channel residual convolution with a nonlinear 48-channel bottleneck will provide enough runtime headroom for the proven 0.995 EMA to finish and exceed 9,214 correct predictions while retaining 249,854 learned parameters.
change: Preserve batch-32 flip-paired matched-view training, replace the residual branch with a cheaper bottleneck, widen the classifier to 144 units, maintain per-step EMA parameters for final evaluation, and batch each mirrored inference pair.
mechanism: Bottleneck-funded exact parameter EMA
evidence_used: The current full-width design completed with 9,214 correct, while exact 0.995 EMA reached 9,241 but timed out; modest width reductions also timed out, motivating a larger targeted reduction in the dominant residual convolution rather than weakening the established training rule.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch-32 dense optimization with matched cardinal augmentation and flip-paired supervision restricted to the final quarter will finish within the time limit and exceed 9,214 correct predictions.
change: Use Reference Design 3’s batch size, learning rate, and five inference-matched spatial views, while limiting duplicated flip forwards to the final 25% of updates.
mechanism: Compute-capped late flip-pairing curriculum
evidence_used: Reference Design 3 completed with 9,214 correct using batch-32 flip pairing throughout, while a single-view-first, paired-second-half variant timed out; quarter-length late pairing reduces training forwards from 150,000 to 125,000 while retaining late exact symmetry.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Halving Reference Design 3’s training forwards while preserving its 3,126 optimizer updates, balanced flip exposure, and the previously accuracy-improving 0.995 EMA will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt the 249,550-parameter residual model, batch-32 matched cardinal augmentation, balanced single-view flips, pair-batched ten-view inference, and fused per-step EMA installed for final evaluation.
mechanism: Single-forward stratified symmetry with exact parameter EMA
evidence_used: Reference Design 3 achieved 9,214 correct but required 81.3 seconds, while exact 0.995 EMA reached 9,241 correct before timing out at 82.89 seconds; eliminating duplicated flip forwards reduces training exposure from 200,000 to 100,000 images while retaining dense optimization and EMA stabilization.
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
