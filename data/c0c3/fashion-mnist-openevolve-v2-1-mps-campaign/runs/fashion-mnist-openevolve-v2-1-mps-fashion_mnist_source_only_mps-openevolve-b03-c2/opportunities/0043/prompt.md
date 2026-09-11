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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 247538, "training_seconds": 22.307899000123143, "validation_accuracy": 0.9105, "validation_correct": 9105, "validation_cross_entropy": 0.25879054794311523, "validation_score": 9105.39720666859}
prior_hypothesis: Expanding convolutional capacity to 247,538 parameters, doubling optimizer updates via batch size 128, and using one augmented pass followed by clean fine-tuning will exceed 8,928 correct validation predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249550, "training_seconds": 63.629130667075515, "validation_accuracy": 0.9119, "validation_correct": 9119, "validation_cross_entropy": 0.29910097885131837, "validation_score": 9119.384881551272}
prior_hypothesis: Ten-view inference over centered and cardinal shifts with horizontal mirrors will exceed 9,112 correct predictions while preserving ample runtime headroom.

REFERENCE DESIGN 3
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 81.30000304197893, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.24333785018920898, "validation_score": 9214.402143311188}
prior_hypothesis: Replacing ±2-pixel and diagonal training crops with the exact five spatial views used by ten-view inference will exceed 9,208 correct predictions without increasing training time.

## Recent verification evidence

RECENT RESULT
hypothesis: Halving Reference Design 3’s training forwards while preserving its 3,126 optimizer updates, balanced flip exposure, and the previously accuracy-improving 0.995 EMA will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt the 249,550-parameter residual model, batch-32 matched cardinal augmentation, balanced single-view flips, pair-batched ten-view inference, and fused per-step EMA installed for final evaluation.
mechanism: Single-forward stratified symmetry with exact parameter EMA
evidence_used: Reference Design 3 achieved 9,214 correct but required 81.3 seconds, while exact 0.995 EMA reached 9,241 correct before timing out at 82.89 seconds; eliminating duplicated flip forwards reduces training exposure from 200,000 to 100,000 images while retaining dense optimization and EMA stabilization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Balanced single-forward flip exposure plus a 16-step approximation of the accuracy-improving 0.995 EMA will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt Reference Design 3’s residual network, batch-32 matched cardinal-view curriculum, and ten-view inference; replace duplicated flip forwards with balanced within-batch flips and apply parameter EMA only every 16 updates.
mechanism: Compute-funded sparse parameter EMA
evidence_used: Reference Design 3 completed with 9,214 correct, while exact 0.995 EMA reached 9,241 but timed out; using one training forward and one-sixteenth as many EMA updates targets the demonstrated gain with substantially less computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling a 0.995-equivalent EMA every 16 updates and replacing ten-view inference with center-plus-mirror inference will finish within the time limit and exceed 9,214 correct predictions.
change: Maintain a low-overhead EMA of learned parameters, install it after the final update, and evaluate only the batched centered image and its mirror.
mechanism: Two-view inference funded sparse parameter EMA
evidence_used: Exact 0.995 EMA reached 9,241 correct, 27 above the current 9,214, while shifted ten-view inference improved Reference Design 3 by only seven correct; removing eight inference views should fund EMA with a smaller expected accuracy cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying the demonstrated 0.995 per-step parameter EMA to the completed batch-64 design will finish within the time limit and exceed 9,119 correct predictions by suppressing late-update noise.
change: Preserve the current architecture, training, augmentation, schedule, and ten-view inference; maintain a fused EMA of all learned parameters and install it after the final optimizer update.
mechanism: Runtime-headroom-funded exact parameter EMA
evidence_used: Exact 0.995 EMA previously raised the strongest batch-32 result from 9,214 to 9,241 correct with about 1.6 seconds of added runtime, while the current design completed in 63.63 seconds and therefore has substantially more headroom.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249550, "training_seconds": 68.52765083312988, "validation_accuracy": 0.9093, "validation_correct": 9093, "validation_cross_entropy": 0.3017596038818359, "validation_score": 9093.384095495443}

RECENT RESULT
hypothesis: A 32-channel nonlinear residual bottleneck will fund batch-32 flip-paired training, while a 2%-floor cosine schedule suppresses late-update noise and exceeds 9,214 correct predictions within the time limit.
change: Replace the current network with a 249,971-parameter bottleneck residual model, use inference-matched cardinal crops, supervise both horizontal orientations, add ten-view inference, and lower the terminal learning rate to 4e-5.
mechanism: Bottleneck-funded paired-view stabilization
evidence_used: Reference Design 3 achieved 9,214 correct with dense flip-paired optimization; 0.995 EMA reached 9,241 before timing out, indicating useful late-weight stabilization, while the prior 48-channel bottleneck attempt showed that additional compute reduction is needed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Deterministically balancing mirrored and unmirrored samples within every batch will retain useful flip invariance and 3,126 dense optimizer updates while eliminating duplicated training forwards, finishing comfortably and exceeding 9,214 correct predictions through less-correlated gradient exposure.
change: Adopt the proven 249,550-parameter residual network, batch-32 cardinal-view curriculum, cosine schedule, and ten-view inference, but replace flip-paired supervision with alternating within-batch flips.
mechanism: Single-forward stratified flip augmentation
evidence_used: Reference Design 3 achieved 9,214 correct but required 81.3 seconds with two training views per example; prior single-forward attempts also added EMA, which repeatedly caused timeouts, so isolated single-forward stratified symmetry without EMA is the clearest untested compute-saving change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A five-checkpoint approximation of the accuracy-improving 0.995 EMA, combined with pair-batched inference, will finish within the time limit and exceed 9,214 correct predictions.
change: Accumulate five strategically weighted late-training parameter snapshots, install their EMA-kernel approximation after the final update, and evaluate each original/mirrored view pair in one forward pass.
mechanism: Exponential-kernel checkpoint weight averaging
evidence_used: Exact 0.995 EMA reached 9,241 correct but timed out from continuous averaging overhead; five checkpoint operations approximate the same temporal weighting at negligible training cost, while Reference Design 3 established pair-batched mirrored inference as valid.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reference Design 3’s dense batch-32 matched-view training with a 2% terminal learning-rate floor will exceed 9,214 correct predictions by reducing late-update noise without adding runtime overhead.
change: Restore the strongest verified batch-32 flip-paired procedure, retain efficient pair-batched ten-view inference, and lower the final learning rate from 2e-4 to 4e-5.
mechanism: Compute-free near-zero cosine stabilization
evidence_used: Reference Design 3 achieved 9,214 correct, while 0.995 EMA reached 9,241 before timing out; this motivates targeting the same late-training instability through a zero-overhead lower cosine floor.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The current 96-channel network can afford Reference Design 3’s batch-32 flip-paired training and ten-view inference within the time limit, and its 3,126 matched-view updates will exceed 9,214 correct predictions.
change: Retain the current 247,538-parameter model, but use batch 32, inference-matched cardinal crops, paired mirror supervision, ten-view inference, and Reference Design 3’s learning rate.
mechanism: Width-funded dense matched-view optimization
evidence_used: Reference Design 3 achieved 9,214 correct with this training rule at 81.3 seconds; the current narrower 24/48/96 convolutional stack completed in 22.3 seconds and provides a compute-saving basis for adopting the stronger procedure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch-40 matched-view training with paired flip consistency for the first 75% of updates and a cheaper clean final quarter will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt the 249,550-parameter residual model and ten-view inference, use 2,500 optimizer steps, add inexpensive flip-consistency regularization, and remove duplicated flip forwards during the final quarter.
mechanism: Front-loaded flip consistency with clean-tail fine-tuning
evidence_used: Reference Design 3 reached 9,214 correct but required 81.3 seconds; Reference Design 1 established clean late fine-tuning as effective, while the batch-40 EMA attempt timed out, motivating fewer training forwards and no parameter averaging.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training uniformly on the five inference views for all 100,000 examples will exceed 9,214 correct predictions by eliminating the current center-view overrepresentation without adding model-forward computation.
change: Extend the deterministic center-and-cardinal crop cycle through the entire training budget instead of switching to clean-only images halfway through.
mechanism: Full-exposure train–inference view matching
evidence_used: The current matched-view design achieved 9,214 correct, while prior evidence found shifted ten-view inference beneficial. Currently 60% of training examples are centered although inference weights all five spatial views equally; this patch makes both distributions 20% per view.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch 48 with 2,084 dense updates and exact cardinal-view augmentation will finish within the runtime limit and exceed the current 9,119 correct predictions.
change: Reduce batch size from 64 to 48, interpolate the learning rate to 2.25e-3, and replace the broader ±2-pixel crop grid with the five views used at inference.
mechanism: Runtime-safe intermediate-batch matched-view optimization
evidence_used: Reference Design 3 reached 9,214 correct using batch-32 dense optimization and matched cardinal views, while the current batch-64 design finishes in 63.63 seconds; batch 48 captures additional updates and view matching without the timeout-prone 3,126-step regime.
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
