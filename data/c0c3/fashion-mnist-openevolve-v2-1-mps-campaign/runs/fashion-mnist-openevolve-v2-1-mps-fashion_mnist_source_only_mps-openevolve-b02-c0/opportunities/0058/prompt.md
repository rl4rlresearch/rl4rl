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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.91783266700804, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.22355818634033203, "validation_score": 9252.40864423579}
prior_hypothesis: Pair-batching mirrored views will preserve the ten-view ensemble while reducing evaluation overhead enough to complete verification, and scaling its logits by 1.10 will preserve 9,252 correct predictions while lowering cross-entropy below 0.2287977.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling the ensemble log-probabilities by 1.10 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2287977, and removing the unused evaluation forward pass will allow the previously timing-out calibration test to complete.
change: Avoid computing discarded unaugmented logits during evaluation and increase the inference-only ensemble scale from 1.05 to 1.10.
mechanism: Efficient post-ensemble logit sharpening
evidence_used: Scaling from 1.00 to 1.05 preserved 9,252 correct and lowered cross-entropy from 0.2356299 to 0.2287977; subsequent sharpening trials timed out, while the current evaluation path performs one full model pass whose logits are discarded.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Pair-batching mirrored views will preserve the ten-view ensemble while reducing evaluation overhead enough to complete verification, and scaling its logits by 1.10 will preserve 9,252 correct predictions while lowering cross-entropy below 0.2287977.
change: Remove the discarded evaluation pass, evaluate each view and its mirror in one batched forward pass, and raise the ensemble scale from 1.05 to 1.10.
mechanism: Batched test-time augmentation with conservative logit sharpening
evidence_used: Scaling from 1.00 to 1.05 preserved 9,252 correct and reduced cross-entropy from 0.2356299 to 0.2287977; later sharpening tests timed out, and removing only the single discarded pass was insufficient, motivating more substantial evaluation batching without changing the mathematical ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.91783266700804, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.22355818634033203, "validation_score": 9252.40864423579}

RECENT RESULT
hypothesis: Increasing the positive ensemble scale from 1.10 to 1.20 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2235582.
change: Raise the inference-only scale applied to the unchanged ten-view ensemble log-probabilities from 1.10 to 1.20.
mechanism: Post-ensemble logit sharpening
evidence_used: With pair-batched evaluation, increasing the scale from 1.05 to 1.10 preserved all 9,252 predictions and reduced cross-entropy from 0.2287977 to 0.2235582; positive scaling cannot change the ensemble argmax, and the continued cross-entropy improvement indicates remaining underconfidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batching the ten unchanged views into two forward passes will avoid the timeout, while scaling their ensemble log-probabilities to 1.20 will preserve 9,252 correct predictions and reduce cross-entropy below 0.2235582.
change: Replace five pair-batched evaluation passes with two five-view stacked passes and raise the inference-only ensemble scale from 1.10 to 1.20.
mechanism: Five-view stacked test-time augmentation with stronger logit calibration
evidence_used: Pair batching completed with 9,252 correct and reduced cross-entropy at scale 1.10; the otherwise unchanged 1.20 trial timed out, so further evaluation batching directly targets completion while preserving the mathematical ensemble.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Grouping two spatial views and their mirrors per forward pass will complete within the verification limit while preserving the ten-view ensemble, and scaling its log-probabilities by 1.20 will preserve 9,252 correct predictions while reducing cross-entropy below 0.2235582.
change: Replace five pair-batched evaluation passes with three moderately sized grouped passes and raise the inference-only ensemble scale from 1.10 to 1.20.
mechanism: Moderate-stacked test-time augmentation with stronger logit calibration
evidence_used: Pair batching completed successfully at scale 1.10 with 9,252 correct and 0.2235582 cross-entropy, while the more aggressive five-view stacking timed out; two-view grouping tests an intermediate throughput/memory tradeoff, and positive scaling cannot change the ensemble argmax.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the scale from 1.10 to 1.15 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2235582.
change: Increase only the positive scale applied to the unchanged pair-batched ten-view ensemble log-probabilities.
mechanism: Midpoint post-ensemble logit sharpening
evidence_used: Scaling from 1.05 to 1.10 preserved all 9,252 predictions and lowered cross-entropy from 0.2287977 to 0.2235582; positive scaling cannot alter argmax, while 1.15 with the successful pair-batched evaluation path remains untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting the center view at 5/13 and each shifted view at 2/13, matching their training-time sampling frequencies, will increase validation_correct above 9,252 while retaining the successful 1.10 calibration scale.
change: Replace equal weighting of the ten test-time views with probability-space weights matching the spatial augmentation distribution; mirrored and unmirrored variants split each spatial weight equally.
mechanism: Augmentation-distribution-matched ensemble weighting
evidence_used: The best design uses equal-weight ten-view averaging, but its training augmentation samples the center crop 5/13 of the time and each directional shift 2/13; aligning inference weights with this observed training distribution is the most direct accuracy-oriented change.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining every spatial view but omitting the four redundant shifted mirrors will avoid recent verification timeouts, while center-weighted aggregation will increase validation_correct above 9,252.
change: Reduce evaluation from ten views in five forward passes to six views in three forward passes, weighting the center and shifted views according to their training-time sampling probabilities while retaining the successful 1.10 scale.
mechanism: Lean distribution-matched test-time ensemble
evidence_used: The ten-view pair-batched design achieved 9,252 correct, whereas subsequent equally expensive calibration and weighting trials repeatedly timed out; the uncompleted weighting trial also identified the center’s 5/13 training frequency as the strongest accuracy-oriented aggregation signal.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the batch size from 48 to 64 will provide enough training and evaluation throughput to avoid recent timeouts while preserving at least 9,252 correct predictions through less noisy gradients and batch-normalization estimates.
change: Increase only the batch size, retaining the successful architecture, optimizer schedule, augmentation, late averaging, ten-view ensemble, and 1.10 calibration scale.
mechanism: Larger-minibatch throughput with improved batch-normalization statistics
evidence_used: The available design reached 9,252 correct but required 76.9 training seconds, and numerous mathematically minor inference changes subsequently timed out; reducing optimizer steps from 2,084 to approximately 1,563 directly targets this recurring constraint with minimal algorithmic disruption.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging view logits instead of probabilities will exceed 9,252 correct predictions by preventing a single overconfident transformed view from dominating, while removing ten log-softmax operations will slightly reduce evaluation time.
change: Replace probability-space aggregation of the unchanged ten pair-batched views with arithmetic logit averaging, retaining the successful 1.10 calibration scale.
mechanism: Geometric-mean test-time ensembling
evidence_used: The pair-batched ten-view ensemble completed with 9,252 correct, and stronger positive scaling improved cross-entropy, indicating underconfident aggregate outputs; logit averaging produces a naturally sharper geometric-mean consensus and requires less evaluation work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Lowering label smoothing from 0.03 to 0.01 will increase validation_correct above 9,252 by strengthening class-discriminative gradients during the fixed two exposures while retaining enough regularization for generalization.
change: Reduce only the training loss label-smoothing coefficient, leaving the successful architecture, schedule, augmentation, late averaging, ten-view ensemble, and 1.10 calibration unchanged.
mechanism: Reduced target smoothing for faster class separation
evidence_used: The successful ensemble remained underconfident—raising its scale from 1.05 to 1.10 preserved 9,252 predictions while lowering cross-entropy—so the current 0.03-smoothed targets are a grounded training-side source of suppressed confidence; repeated inference-only changes have supplied no accuracy gain.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding parameter-free residual paths across each same-width convolutional pair will raise validation_correct above 9,252 by improving optimization during the fixed two exposures.
change: Replace the sequential feature forward with two residual convolutional stages while preserving all layers, parameters, augmentation, optimizer, and inference ensemble.
mechanism: Intra-stage residual feature learning
evidence_used: The current architecture reached 9,252 correct, while inference-only calibration changes cannot improve its argmax and subsequent loss changes yielded no completed accuracy evidence; residual paths directly target faster feature learning without increasing parameters or model passes.
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
