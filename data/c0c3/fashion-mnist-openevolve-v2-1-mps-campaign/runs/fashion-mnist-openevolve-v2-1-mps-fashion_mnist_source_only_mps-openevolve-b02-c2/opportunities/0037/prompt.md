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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.76099287485704, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19212254104614257, "validation_score": 9312.419419969663}
prior_hypothesis: Replacing smooth maximum attention with the strongest 2×2 activation average will exceed 9,322 correct predictions by capturing coherent four-activation evidence without top-k’s runtime cost or regional pooling’s forced spatial coverage.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.78843287518248, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.1926749450683594, "validation_score": 9307.419225709458}
prior_hypothesis: Training the proven 9,320-correct shared average-plus-maximum attention model with a balanced individual-view and probability-ensemble loss will exceed 9,322 correct predictions by directly optimizing the same arithmetic flip ensemble used during validation while retaining supervision for both orientations.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging paired-orientation logits will exceed 9,322 correct predictions by favoring cross-orientation consensus over a single view’s disproportionately confident probability.
change: Replace probability-space flip ensembling with arithmetic logit averaging; training, parameters, and forward-pass count remain unchanged.
mechanism: Geometric flip-ensemble inference
evidence_used: Paired-orientation supervision and flip-ensemble validation were beneficial, while adding Jensen–Shannon alignment timed out; logit averaging provides a computationally free consistency bias at inference.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing hard channel maxima with an unsorted top-four mean will exceed the current 9,320 correct predictions while avoiding unnecessary sorting overhead from the 9,322-correct reference implementation.
change: Use the mean of each channel’s four strongest spatial activations as the shared salient descriptor, with `sorted=False` to reduce top-k runtime; preserve all other training and inference behavior.
mechanism: Unsorted top-four channel saliency
evidence_used: Top-four channel attention achieved the best verified result at 9,322 correct versus 9,320 for hard maxima, but later top-k variants timed out, motivating the same validated statistic with sorting disabled because only its mean is consumed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training the proven 9,320-correct shared average-plus-maximum attention model with a balanced individual-view and probability-ensemble loss will exceed 9,322 correct predictions by directly optimizing the same arithmetic flip ensemble used during validation while retaining supervision for both orientations.
change: Restore shared global-average-plus-maximum channel attention and replace plain paired-view cross-entropy with an equal blend of individual-view cross-entropy and validation-matched flip-ensemble negative log-likelihood.
mechanism: Probability-ensemble-aware flip-pair training
evidence_used: Shared average-plus-maximum attention reached 9,320 correct and paired-view probability ensembling was beneficial; top-four attention improved by only two additional correct predictions but repeatedly encountered runtime failures, motivating a computationally light objective-level refinement on the reliable 9,320-correct design.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 72.78843287518248, "validation_accuracy": 0.9307, "validation_correct": 9307, "validation_cross_entropy": 0.1926749450683594, "validation_score": 9307.419225709458}

RECENT RESULT
hypothesis: Replacing hard channel maxima with a temperature-0.25 log-mean-exp descriptor will exceed 9,322 correct predictions by approximating robust top-four saliency with dense gradients and without top-k runtime overhead.
change: Remove the harmful translation augmentation and replace global-maximum channel evidence with a parameter-free smooth-maximum descriptor while preserving shared attention, paired-view training, and flip-ensemble validation.
mechanism: Temperature-controlled smooth channel saliency
evidence_used: Top-four channel attention achieved the best verified result at 9,322 correct versus 9,320 for hard maxima, but subsequent top-k variants timed out; translations reduced correctness to 9,178. A smooth reduction tests the same robust-saliency principle without sorting.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.23836104106158, "validation_accuracy": 0.9301, "validation_correct": 9301, "validation_cross_entropy": 0.19321219635009765, "validation_score": 9301.419036950452}

RECENT RESULT
hypothesis: Detaching only the top-four descriptor from backbone gradients will exceed 9,322 correct predictions by retaining the successful forward gate while preventing sparse selected-location gradients from destabilizing feature learning.
change: Compute the identical top-four activation mean from detached features; preserve all forward behavior, parameters, training loss, and inference ensembling.
mechanism: Stop-gradient top-four channel saliency
evidence_used: Shared top-four attention achieved the best result at 9,322 correct, while the dense-gradient log-mean-exp alternative fell to 9,301 and subsequent top-k experiments repeatedly timed out; this isolates salient-descriptor backpropagation while modestly reducing backward work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging four regional spatial maxima per channel will exceed 9,322 correct predictions by retaining robust multi-activation evidence without the repeated top-k operations that timed out.
change: Replace each channel’s single global maximum with the mean of a 2×2 grid of regional maxima; preserve shared attention, paired-view cross-entropy, and flip-ensemble validation.
mechanism: Spatially distributed regional-max channel attention
evidence_used: Top-four channel saliency improved global-maximum attention from 9,320 to 9,322 correct, but later top-k variants repeatedly timed out; regional max pooling provides four salient values through a single pooling operation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 71.30843037506565, "validation_accuracy": 0.9301, "validation_correct": 9301, "validation_cross_entropy": 0.19263489723205565, "validation_score": 9301.419239786761}

RECENT RESULT
hypothesis: Restoring plain paired-view cross-entropy and averaging sparse low-learning-rate tail checkpoints will exceed 9,322 correct predictions by recovering the proven 9,320-correct objective while reducing late-training parameter noise.
change: Remove the harmful ensemble-aware loss and install a running average sampled every 20 optimizer steps during the final 20% of training, copying it into the model after the last step.
mechanism: Sparse tail weight averaging on reliable maximum attention
evidence_used: Plain paired-view cross-entropy with maximum attention achieved 9,320 correct, while the ensemble-aware objective reduced that architecture to 9,307. Top-four attention added only two correct predictions and repeatedly approached or exceeded the runtime limit, motivating a low-overhead optimization refinement of the reliable maximum-attention design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing normalized log-mean-exp with a temperature-0.25 softmax-weighted activation mean will exceed 9,322 correct predictions by retaining dense gradients while restoring the peak-scale descriptor lost through log-mean-exp normalization.
change: Use the Gibbs-weighted mean of spatial activations as the salient channel descriptor, preserving architecture, parameters, training, and flip-ensemble inference.
mechanism: Softmax-weighted channel saliency
evidence_used: Top-four saliency achieved 9,322 correct and hard maxima achieved 9,320, while normalized log-mean-exp fell to 9,301; softmax weighting preserves the current efficient dense selection but avoids the normalization-induced downward shift in descriptor magnitude.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.03723650006577, "validation_accuracy": 0.9299, "validation_correct": 9299, "validation_cross_entropy": 0.19349568481445312, "validation_score": 9299.418937417506}

RECENT RESULT
hypothesis: Blending 25% hard-maximum evidence into the top-four descriptor will exceed 9,322 correct predictions by preserving top-four robustness while recovering useful peak emphasis.
change: Reweight the existing top-four activations toward their strongest member without adding parameters or another top-k operation.
mechanism: Peak-biased top-four channel saliency
evidence_used: Top-four averaging achieved 9,322 correct, while hard maximum achieved 9,320 with slightly lower cross-entropy (0.192262 versus 0.192551), motivating a conservative interpolation between them.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A temperature-2 interpolation between probability averaging and logit averaging will exceed 9,322 correct predictions by rewarding cross-orientation consensus without fully adopting the unverified geometric endpoint.
change: Keep the reliable 9,320-correct model and training procedure unchanged, but replace arithmetic probability ensembling with a temperature-2 power mean during validation.
mechanism: Tempered power-mean flip ensembling
evidence_used: Arithmetic probability ensembling reached 9,320 correct, while ensemble-aware training reduced this to 9,307 and pure logit averaging timed out; an inference-only interpolation isolates ensemble geometry without adding training work.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 80.11578466696665, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.19228479461669923, "validation_score": 9320.419362892371}

RECENT RESULT
hypothesis: Restoring plain paired-view cross-entropy with 0.02 label smoothing will exceed 9,322 correct predictions by preserving the reliable maximum-attention training signal while modestly regularizing overconfident decisions.
change: Remove the harmful ensemble-aware objective and train both orientations with lightly smoothed cross-entropy.
mechanism: Mild paired-view label smoothing
evidence_used: Plain paired-view cross-entropy with maximum attention reached 9,320 correct, while the ensemble-aware loss reduced the same architecture to 9,307; this isolates a low-overhead regularization change on the reliable design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing smooth maximum attention with the strongest 2×2 activation average will exceed 9,322 correct predictions by capturing coherent four-activation evidence without top-k’s runtime cost or regional pooling’s forced spatial coverage.
change: Replace temperature-controlled log-mean-exp saliency with a parameter-free maximum over local 2×2 average-pooled responses; preserve shared attention, paired-view training, and probability flip ensembling.
mechanism: Strongest-local-patch channel saliency
evidence_used: Arbitrary top-four averaging achieved the best result at 9,322 correct, while hard maxima reached 9,320 and four fixed regional maxima fell to 9,301; selecting the strongest local four-value patch tests robust multi-activation evidence without requiring irrelevant regions or expensive top-k selection.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 76.76099287485704, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19212254104614257, "validation_score": 9312.419419969663}



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
