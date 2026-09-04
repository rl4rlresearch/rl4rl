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

RECENT RESULT
hypothesis: Widening the regularized classifier bottleneck from 48 to 56 units will increase validation_correct above 9,252 while remaining below the parameter ceiling.
change: Use the available parameter headroom to expand both classifier linear layers and their intervening batch normalization, raising learned parameters from 216,346 to 241,538.
mechanism: Expanded spatial classification bottleneck
evidence_used: The verified model leaves 33,654 parameters unused, while repeated inference-only changes cannot improve argmax accuracy; expanding the narrow 48-unit projection of 3,136 spatial features directly increases class-discriminative capacity with little added runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Folding evaluation-time batch normalization into adjacent convolutions and the classifier projection will prevent verification timeout while preserving 9,252 predictions, and a 1.20 ensemble scale will reduce cross-entropy below 0.2235582.
change: Fuse batch-normalization parameters into cached inference-only weights, retain the complete ten-view ensemble, and raise its calibration scale from 1.10 to 1.20.
mechanism: Batch-normalization folding for fast calibrated ten-view inference
evidence_used: The 1.10 ten-view design completed with 9,252 correct, while scaling from 1.05 to 1.10 lowered cross-entropy without changing predictions; subsequent equally expensive inference variants timed out, motivating an exactly equivalent but cheaper evaluation path.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing evaluation from ten augmented views to the center view and its mirror will complete verification reliably while retaining at least 9,200 correct predictions and comparable calibration.
change: Replace the five-pass ten-view ensemble with one pair-batched forward pass over the original images and their horizontal mirrors, retaining the verified 1.10 scale.
mechanism: Lean center-view flip ensemble
evidence_used: The ten-view pair-batched design reached 9,252 correct, but every subsequent design timed out, including a six-view variant; cutting evaluation workload from 100,000 to 20,000 image views directly addresses the recurring completion failure while preserving the training-aligned flip ensemble.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Eliminating test-time augmentation will finish within the verification limit while retaining at least 9,100 correct predictions, establishing whether evaluation overhead caused the repeated timeouts.
change: Replace the ten-view, five-pass evaluation ensemble with one calibrated center-view forward pass; training remains unchanged.
mechanism: Single-view inference fast path
evidence_used: The verified ten-view design reached 9,252 correct but took 76.9 training seconds, and every subsequent variant timed out—including the two-view center/flip ensemble—so halving that latest evaluation workload is the most direct completion-oriented test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the unchanged ten-view ensemble by 1.20 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2235582.
change: Increase only the positive inference-time ensemble scale from 1.10 to 1.20.
mechanism: Isolated post-ensemble calibration sharpening
evidence_used: Raising the scale from 1.05 to 1.10 preserved all 9,252 predictions and reduced cross-entropy from 0.2287977 to 0.2235582; prior 1.20 trials also changed evaluation mechanics, so this isolates calibration on the only verified-completing path.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing convolution widths to 24/48 while widening the classifier bottleneck to 80 will finish reliably by cutting convolutional work roughly 44%, while retaining at least 9,252 correct predictions through a larger spatial classification head.
change: Reallocate capacity from expensive feature-map convolutions to the classifier, producing a 226,002-parameter model while preserving the verified training procedure and ten-view ensemble.
mechanism: Compute-aware capacity reallocation
evidence_used: Single-view inference and batch-size 64 both still timed out, indicating evaluation overhead and optimizer-step count alone were not the primary issue; reducing training-time convolutional computation directly targets the remaining bottleneck.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A low-cost bottleneck residual block plus a 54-unit classifier will increase validation_correct above 9,252 while remaining below the 250,000-parameter ceiling.
change: Add an identity-initialized 64→32→64 residual block at 7×7 resolution and widen the classifier bottleneck from 48 to 54 units, yielding 248,808 learned parameters.
mechanism: Identity-initialized post-pool residual refinement
evidence_used: The verified design reached 9,252 correct with 33,654 parameters unused; inference-only changes cannot improve its argmax, while previous training-side trials timed out without providing contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing per-image indexed crops with balanced batch-shared crops will finish verification while retaining at least 9,252 correct predictions because it preserves the 5:2:2:2:2 translation exposure distribution and removes costly advanced indexing from every training step.
change: Use the training-step index to cycle through the same thirteen translation outcomes, applying each batch’s translation with a contiguous slice while preserving independent horizontal flips and the verified model and ensemble.
mechanism: Stratified batch-shared translation
evidence_used: The verified design achieved 9,252 correct but took 76.9 training seconds; even single-view evaluation timed out later, pointing to training-path cost, while prior compute reductions left the per-example advanced-index augmentation unchanged.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.19069349998608, "validation_accuracy": 0.9218, "validation_correct": 9218, "validation_cross_entropy": 0.22453779983520508, "validation_score": 9218.408317325988}



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
