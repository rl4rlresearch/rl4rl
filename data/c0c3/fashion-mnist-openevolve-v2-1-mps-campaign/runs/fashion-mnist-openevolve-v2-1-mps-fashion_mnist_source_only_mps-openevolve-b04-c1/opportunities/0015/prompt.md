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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 63.53062425018288, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.21036614379882812, "validation_score": 9286.413098137751}
prior_hypothesis: Increasing the EMA interpolation rate from 0.01 to 0.02 will exceed 9,282 correct predictions by reducing parameter lag while retaining useful smoothing over roughly the final 50 optimizer steps.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Blending 20% individual-view cross-entropy into the proven paired-logit objective will exceed 9,282 correct predictions by discouraging view-specific errors while retaining direct optimization of the validation ensemble.
change: Compute cross-entropy for both original and mirrored logits and blend it with the existing ensemble loss at an 80/20 ratio.
mechanism: Auxiliary per-view supervision for paired ensemble training
evidence_used: Paired-view ensemble training improved validation correct from 9,237 to 9,282; auxiliary supervision is a focused test of whether ensuring both constituent views remain discriminative can further strengthen that successful predictor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 65.5410473330412, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2114208724975586, "validation_score": 9252.412738472112}

RECENT RESULT
hypothesis: Replacing fixed max pooling and the oversized dense head with residual strided-convolution downsampling and a third convolutional stage will exceed 9,282 correct predictions by learning which spatial information to preserve at each resolution.
change: Challenge the assumption that fixed max pooling plus a high-dimensional flattened head is the best capacity allocation; use learned residual downsampling, progressive 32→48→72 channels, and a compact classifier while retaining the proven paired-view training and EMA.
mechanism: Residual learned-downsampling feature hierarchy
evidence_used: Simply deepening the existing max-pooled branches regressed from 9,212 to 9,196, and translation ensembling fell to 9,187, suggesting that neither more same-resolution convolutions nor inference-time correction solves the representation bottleneck. The 9,282-result paired objective remains intact while this patch tests a different spatial computation within the parameter ceiling.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the algebraically redundant pre-pair horizontal flip with random one-pixel translations will exceed 9,282 correct predictions by teaching translation robustness while preserving the proven mirrored-pair objective.
change: Sample the center or one of four cardinal one-pixel translations independently per training image using replicated-edge padding; paired mirror training, architecture, optimizer, and EMA remain unchanged.
mechanism: Paired-view translation augmentation
evidence_used: Paired-view training reached 9,282 correct, but its existing random pre-flip only swaps the two paired views and adds no augmentation. Translation ensembling regressed to 9,187, indicating shifted predictions are not yet reliable; training on those shifts directly tests whether learned invariance can recover their potential without increasing forward-pass count.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 64.92469508294016, "validation_accuracy": 0.9215, "validation_correct": 9215, "validation_cross_entropy": 0.21804719619750976, "validation_score": 9215.41049312503}

RECENT RESULT
hypothesis: Averaging learned parameters while retaining the latest BatchNorm running statistics will exceed 9,282 correct predictions by avoiding normalization-buffer lag in the small-batch EMA model.
change: Track parameter names and apply EMA only to learned parameters; copy BatchNorm buffers directly from the latest training state.
mechanism: Parameter-only EMA with fresh BatchNorm statistics
evidence_used: EMA at rate 0.01 improved validation correct from 9,215 to 9,237, while the longer 0.005 horizon regressed to 9,233, suggesting excessive state lag is harmful; paired training subsequently raised the same EMA design to 9,282.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 44.262633834034204, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.209847159576416, "validation_score": 9282.413275343122}

RECENT RESULT
hypothesis: Halving the batch size to 32 with square-root learning-rate scaling will exceed 9,282 correct predictions by doubling optimizer updates while the paired-view forward pass still supplies 64 images per BatchNorm update.
change: Reduce batch size from 64 to 32 and scale both the initial and scheduled AdamW learning rates from 2.1e-3 to 1.5e-3.
mechanism: Finer-grained paired-view optimization
evidence_used: The prior 128-to-64 batch reduction increased validation correct from 9,212 to 9,215 with the same 216,250-parameter architecture; applying the same successful optimization change to the stronger paired-view, parameter-only-EMA design tests whether additional update frequency yields further gains.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the EMA interpolation rate from 0.01 to 0.02 will exceed 9,282 correct predictions by reducing parameter lag while retaining useful smoothing over roughly the final 50 optimizer steps.
change: Shorten the learned-parameter EMA horizon from approximately 100 steps to 50 steps, leaving the architecture, paired-view objective, BatchNorm buffers, and schedule unchanged.
mechanism: Short-horizon parameter-only EMA
evidence_used: Parameter-only EMA matched the best 9,282 correct while lowering cross-entropy, whereas lengthening the EMA horizon from 0.01 to 0.005 previously reduced correct predictions from 9,237 to 9,233; this directly motivates testing less trajectory lag.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 63.53062425018288, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.21036614379882812, "validation_score": 9286.413098137751}



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
