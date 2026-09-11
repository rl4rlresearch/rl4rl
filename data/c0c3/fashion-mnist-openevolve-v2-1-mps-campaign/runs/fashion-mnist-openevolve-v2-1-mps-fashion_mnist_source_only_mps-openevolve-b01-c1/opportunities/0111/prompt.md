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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 60.70090966694988, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21245321807861328, "validation_score": 9257.412387045162}
prior_hypothesis: Widening the second and residual convolutional stages while shrinking the coordinate-specific classifier will exceed 9,247 correct predictions by learning richer spatial features without adding depth or substantially increasing parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Widening only the residual output from 56 to 60 channels while reducing the classifier from 64 to 60 units will exceed 9,257 correct predictions by adding spatial feature capacity without the over-reallocation of the unsuccessful 64-channel design.
change: Change the residual and shortcut output width to 60 and the flattened classifier width to 60, yielding 239,966 learned parameters.
mechanism: Conservative residual-width reallocation
evidence_used: Reallocating capacity toward convolutional features improved correctness from 9,247 to 9,257, but the more aggressive 44→64-channel design fell to 9,238; this tests a conservative intermediate allocation while retaining the proven 40-channel second stage.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding all-to-all self-attention between the 7×7 feature tokens will exceed 9,257 correct predictions by modeling image-wide relationships that additional local convolutional width and depth failed to capture.
change: Insert a four-head residual self-attention layer before the existing flattened classifier, producing 246,314 learned parameters while preserving the established training and ensemble procedure.
mechanism: Content-adaptive global token mixing
evidence_used: Convolutional reallocation improved to 9,257 correct, but further widening fell to 9,238 and an added spatial convolution reached only 9,228; a larger flattened head also regressed to 9,210. This challenges the shared assumption that either more local extraction or more static coordinate mixing is sufficient, using content-dependent global interactions instead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging per-view class probabilities instead of unbounded logits will exceed 9,257 correct predictions by preventing one confidently incorrect crop from dominating the 50-view ensemble.
change: Convert each validation crop and flip prediction to probabilities before the existing full/center-weighted aggregation, then return calibrated log-probabilities as logits.
mechanism: Posterior-space test-time augmentation
evidence_used: Evaluation-only center-weighted aggregation improved the available design to 9,247 correct, showing inference aggregation can improve correctness; posterior averaging tests a complementary robust aggregation rule without changing the proven architecture or training procedure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 61.762589083984494, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.21763913192749024, "validation_score": 9252.410630692533}

RECENT RESULT
hypothesis: Replacing hard max pooling with a learnable max/average mixture will exceed 9,257 correct predictions by reducing shift aliasing while retaining salient garment features.
change: Add a two-parameter mixed-pooling module, initialized 75% toward max pooling, and use it at both downsampling stages; total parameters become 233,436.
mechanism: Learnable max-average downsampling
evidence_used: Evaluation-only crop weighting improved correctness, showing sensitivity to spatial shifts, while classifier dropout and posterior-space aggregation did not improve the widened model; this tests feature-level shift robustness without adding depth or significant capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233436, "training_seconds": 65.43269150005654, "validation_accuracy": 0.9246, "validation_correct": 9246, "validation_cross_entropy": 0.21335608406066894, "validation_score": 9246.412080185337}

RECENT RESULT
hypothesis: Adding a direct global-average feature readout will exceed 9,257 correct predictions by supplying translation-robust garment evidence while preserving the proven coordinate-sensitive classifier.
change: Add a bias-free, zero-initialized global-average classifier whose logits are summed with the existing flattened-head logits, increasing parameters from 233,434 to 233,994.
mechanism: Zero-initialized global-logit residual
evidence_used: The 233,434-parameter widened convolutional model achieved the best result, while image-conditioned channel gating regressed to 9,204 and global self-attention timed out; a lightweight additive readout tests complementary global context without modulating the successful spatial representation or adding costly token mixing.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233994, "training_seconds": 48.028918457916006, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.2120704231262207, "validation_score": 9253.412517284854}

RECENT RESULT
hypothesis: Replacing the coordinate-specific 7×7 classifier with pooled 4×4, 2×2, and global features will exceed 9,257 correct predictions by reducing crop-position sensitivity while retaining the successful widened convolutional backbone.
change: Concatenate three average-pooled feature scales and classify them with a 160-unit head, producing 247,226 learned parameters.
mechanism: Capacity-matched multiscale spatial pyramid head
evidence_used: Convolutional widening produced the best result at 9,257 correct, while enlarging the flattened head and adding a separate global-logit branch regressed; the earlier spatial-pyramid implementation was not verifiable, leaving a capacity-corrected version on the stronger backbone untested.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the batch size to 96 while scaling the learning rate by 0.75 will exceed 9,257 correct predictions by providing 1,042 less-correlated optimizer updates without increasing cumulative learning-rate exposure.
change: Change batch size from 128 to 96 and scale both the initial and scheduled peak learning rates from 3.0e-4/3.0e-3 to 2.25e-4/2.25e-3.
mechanism: Smaller-batch, linearly scaled AdamW optimization
evidence_used: The 233,434-parameter architecture remains best at 9,257 correct, while further widening, added pooling, dropout, global readout, and alternative aggregation all regressed; this preserves the proven model and tests the largely unexplored optimization granularity axis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 233434, "training_seconds": 76.50440808292478, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2121406379699707, "validation_score": 9252.412493389247}

RECENT RESULT
hypothesis: Rotating half of the training images by up to 8 degrees will exceed 9,257 correct predictions by learning orientation robustness complementary to the existing translation and flip augmentation.
change: Apply per-image bilinear rotation with reflection padding before generating the six translated and flipped training views; leave the architecture and inference ensemble unchanged.
mechanism: Probabilistic mild rotation augmentation
evidence_used: Crop-weighted evaluation demonstrated sensitivity to geometric variation, while further widening, mixed pooling, classifier dropout, and smaller-batch optimization all regressed; mild rotation tests an unexplored geometric invariance without consuming parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 77.94333291705698, "validation_accuracy": 0.9201, "validation_correct": 9201, "validation_cross_entropy": 0.22095044631958008, "validation_score": 9201.409517029546}

RECENT RESULT
hypothesis: Averaging parameters and BatchNorm statistics over the final half of training will exceed 9,257 correct predictions by reducing endpoint noise from the cosine-decayed AdamW trajectory without changing the proven architecture or example exposure.
change: Track a 0.99-decay exponential moving average during the final 50% of optimizer steps and replace the model’s parameters and floating-point buffers with that average after the last step.
mechanism: Tail exponential weight averaging
evidence_used: The 233,434-parameter architecture remains best, while further widening, dropout, pooling, global readouts, alternative aggregation, rotation, and smaller-batch optimization all regressed; tail weight averaging preserves that design and tests optimizer-trajectory stability, an unexplored axis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 74.9380826249253, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.2164536392211914, "validation_score": 9241.411030871937}

RECENT RESULT
hypothesis: Sampling crop offsets independently for every image will exceed 9,257 correct predictions by increasing translation diversity within each optimizer step while preserving the proven augmentation range, architecture, and loss.
change: Replace batch-shared translation offsets with independently sampled per-image offsets for both full-range and central crops, using tensor gathers to construct each view.
mechanism: Per-example stratified translation augmentation
evidence_used: Center-weighted crop evaluation improved the available design, demonstrating useful translation sensitivity, while stronger rotation and altered pooling regressed; increasing diversity within the existing successful translation augmentation tests shift robustness without introducing a new transformation or parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 59.49696479202248, "validation_accuracy": 0.9248, "validation_correct": 9248, "validation_cross_entropy": 0.21336817321777343, "validation_score": 9248.412076079658}

RECENT RESULT
hypothesis: Sampling central training crops with the same separable 1:2:1 weighting used by validation will exceed 9,257 correct predictions by reducing the train–evaluation mismatch in spatial ensemble supervision.
change: Replace uniform sampling of the nine central crop positions with triangular sampling that selects each axis with probabilities 1/4, 1/2, and 1/4.
mechanism: Evaluation-matched triangular crop sampling
evidence_used: Center-weighted validation aggregation previously improved correctness, while independently sampled uniform translations regressed to 9,248; this isolates distribution alignment without changing the proven architecture, augmentation range, or compute cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 66.52478416706435, "validation_accuracy": 0.9255, "validation_correct": 9255, "validation_cross_entropy": 0.21267282524108888, "validation_score": 9255.41231236455}

RECENT RESULT
hypothesis: Adding a normalized channel-covariance classifier will exceed 9,257 correct predictions by capturing position-robust feature co-occurrences that the coordinate-specific flattened head and failed first-order global-average branch cannot represent.
change: Compute the upper-triangular covariance of the 56 feature channels, apply signed-square-root and L2 normalization, and add a zero-initialized ten-class bilinear readout to the existing logits; total learned parameters become 249,404.
mechanism: Residual bilinear covariance pooling
evidence_used: The global-average residual reached only 9,253 correct and additional local depth reached 9,228, suggesting neither first-order pooled evidence nor more local processing supplies the missing discrimination. Full covariance pooling instead introduces second-order channel interactions while preserving the best 233,434-parameter backbone and spatial classifier.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249404, "training_seconds": 57.588185583939776, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21289410247802734, "validation_score": 9249.412237143357}



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
