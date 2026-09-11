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

RECENT RESULT
hypothesis: Preserving local maxima while low-pass filtering before each stride-2 reduction will exceed 9,257 correct predictions by reducing the shift aliasing exposed by crop-weighted validation without sacrificing salient garment features.
change: Add a parameter-free max-then-binomial-blur pooling layer and replace both standard max-pooling stages with it.
mechanism: Anti-aliased max downsampling
evidence_used: Center-weighted crop aggregation improved correctness, indicating downsampling-phase sensitivity, while learned max/average mixing fell to 9,246; anti-aliased max pooling targets that sensitivity while retaining the max operation that the mixture weakened.
result: training did not finish within the verification time limit

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,257 correct predictions by retaining early regularization while allowing late training to sharpen class boundaries for exact top-1 accuracy.
change: Compute a progress-dependent smoothing coefficient and use it consistently in every individual, paired-view, and ensemble cross-entropy term.
mechanism: Cosine-decayed label smoothing
evidence_used: The 233,434-parameter architecture remains best while architectural, augmentation, pooling, averaging, and batch-size changes regressed; the loss schedule is comparatively unexplored, and exact correct count is prioritized over calibration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 76.14576787501574, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.2124201316833496, "validation_score": 9254.412398299017}

RECENT RESULT
hypothesis: Averaging normalized class probabilities across translated and flipped views will exceed 9,257 correct predictions by preventing an overconfident misaligned crop from dominating the ensemble.
change: Replace validation-time arithmetic logit averaging with probability averaging, then return the log-probability mixture with the existing calibration scale.
mechanism: Probability-space test-time augmentation pooling
evidence_used: Center-weighted crop aggregation improved the best design, showing that predictions vary meaningfully across geometric views; probability pooling directly targets that variation without changing the proven training procedure or parameter count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Aligning each translated and flipped prediction with their detached probability ensemble will exceed 9,257 correct predictions by reducing harmful view disagreement without changing the proven transformations or architecture.
change: Retain the existing supervised losses and add a cosine-ramped KL consistency loss between all six view predictions and their 0.9/0.1 full/central probability mixture.
mechanism: Ramped teacher-view consistency regularization
evidence_used: Center-weighted aggregation showed that predictions vary meaningfully across geometric views, while changing the crop distribution or adding rotation regressed; this targets disagreement among the existing successful views instead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Context-dependent channel reweighting will exceed 9,257 correct predictions by exploiting global garment context while preserving the proven coordinate-specific classifier.
change: Add a 960-parameter global-average channel gate after the residual stage, initialized to an exact identity so optimization begins from the current model.
mechanism: Identity-initialized squeeze-and-excitation channel gating
evidence_used: A global-average classification branch reached 9,253 correct, suggesting pooled context contains useful but insufficient standalone evidence; using it to modulate spatial features retains the stronger flattened head.
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
