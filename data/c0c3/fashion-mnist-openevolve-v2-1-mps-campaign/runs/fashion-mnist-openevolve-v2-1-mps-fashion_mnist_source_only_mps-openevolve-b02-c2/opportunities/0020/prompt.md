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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.5085795421619, "validation_accuracy": 0.9178, "validation_correct": 9178, "validation_cross_entropy": 0.2245953540802002, "validation_score": 9178.408298135653}
prior_hypothesis: Restoring the 9,320-correct shared dual-statistic attention model and adding mild per-image translations will exceed 9,320 correct predictions by improving spatial robustness without changing parameter count or the successful paired-view objective.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}
prior_hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 81.16789445909671, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19513330612182617, "validation_score": 9286.418363372051}
prior_hypothesis: Restoring the 9,271-correct attention model and jointly supervising every image in both horizontal orientations will exceed 9,271 correct predictions by aligning training with the successful flip-ensemble evaluation without adding parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the 9,245-correct second bottleneck and adding identity-initialized channel-spatial attention will exceed 9,245 correct predictions while remaining below 250,000 learned parameters.
change: Add Reference Design 2’s second 24-channel residual bottleneck, followed by zero-initialized ECA-style channel attention and spatial attention, for approximately 249,855 parameters.
mechanism: Identity-initialized channel-spatial attention
evidence_used: The second low-resolution bottleneck improved 9,236 to 9,245 correct, whereas EMA reduced accuracy; dynamic feature selection uses the remaining 246-parameter headroom while preserving the best verified optimizer, augmentation, and evaluation regimen.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.02248795796186, "validation_accuracy": 0.9271, "validation_correct": 9271, "validation_cross_entropy": 0.2010115650177002, "validation_score": 9271.416315724648}

RECENT RESULT
hypothesis: Restoring the 9,271-correct attention model and jointly supervising every image in both horizontal orientations will exceed 9,271 correct predictions by aligning training with the successful flip-ensemble evaluation without adding parameters.
change: Add the two verified low-resolution bottlenecks and identity-initialized channel-spatial attention, then compute training cross-entropy on a joint batch containing each prepared image and its mirror.
mechanism: Paired-view flip-symmetric supervision with attention residual refinement
evidence_used: Reference Design 2 achieved the best result at 9,271 correct with attention and flip-ensemble inference; earlier flip-ensemble evaluation improved 9,141 to 9,161, showing that mirrored predictions contain useful complementary signal.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 81.16789445909671, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19513330612182617, "validation_score": 9286.418363372051}

RECENT RESULT
hypothesis: Jointly optimizing individual-view and flip-averaged cross-entropy will exceed 9,286 correct predictions because it directly trains the probability ensemble used during validation while retaining supervision for both orientations.
change: Replace paired-image cross-entropy with an equal-weight combination of per-view cross-entropy and cross-entropy on the averaged original/mirrored probabilities.
mechanism: Ensemble-aligned paired-view supervision
evidence_used: Paired flip supervision improved the attention model from 9,271 to 9,286 correct, and flip-ensemble evaluation previously improved 9,141 to 9,161; directly supervising that ensemble is the next isolated extension of both gains.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 64.902953249868, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.19406703033447265, "validation_score": 9278.418736961408}

RECENT RESULT
hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.
change: Train on both horizontal orientations as in the best design, and compute each channel gate from the sum of global-average and global-maximum descriptors.
mechanism: Dual-statistic channel attention with paired flip supervision
evidence_used: Paired supervision improved the attention model from 9,271 to 9,286 correct, while ensemble-aligned loss regressed to 9,278; attention itself previously improved 9,245 to 9,271, motivating a parameter-neutral refinement of attention while preserving plain paired cross-entropy.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}

RECENT RESULT
hypothesis: Restoring the 9,320-correct dual-statistic attention and paired-view supervision, then reallocating one classifier unit to a parameter-efficient 7×7 residual block, will exceed 9,320 correct predictions while remaining below 250,000 learned parameters.
change: Add the best verified dual-statistic channel-spatial attention and paired flip loss, reduce the classifier from 48 to 47 units, and spend the freed parameters on an identity-initialized depthwise/grouped-convolution residual block.
mechanism: Identity-initialized depthwise-grouped residual refinement
evidence_used: Dual-statistic attention with paired supervision achieved the best result of 9,320 correct, while the earlier additional low-resolution bottleneck improved 9,236 to 9,245; this preserves the winning regimen and tests further low-resolution refinement with approximately 249,588 parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249588, "training_seconds": 74.68625958287157, "validation_accuracy": 0.9261, "validation_correct": 9261, "validation_cross_entropy": 0.19783590965270997, "validation_score": 9261.417419444491}

RECENT RESULT
hypothesis: Adding global channel dispersion to the verified average-plus-maximum attention will exceed 9,320 correct predictions without increasing learned parameters.
change: Preserve the best architecture and training regimen while adding per-channel spatial standard deviation to the shared channel-attention gate.
mechanism: Tri-statistic channel attention
evidence_used: Replacing average-only attention with shared average-plus-maximum evidence improved paired-view validation from 9,286 to 9,320 correct; testing a third complementary statistic is the most isolated extension of that gain.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring shared global-average and global-maximum channel evidence will raise validation correctness from 9,286 toward the verified 9,320 result without adding parameters or costly dispersion computation.
change: Replace average-only channel attention with the verified parameter-neutral average-plus-maximum gate while preserving paired-view supervision and flip-ensemble inference.
mechanism: Dual-statistic channel attention
evidence_used: The identical 249,855-parameter regimen achieved 9,320 correct with average-plus-maximum attention versus 9,286 with average-only attention; adding standard deviation later exceeded the verification time limit.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the 9,320-correct architecture and giving global-average and global-maximum descriptors independent channel-attention kernels will exceed 9,320 correct predictions while remaining below 250,000 parameters.
change: Remove the regressive depthwise-grouped residual block, restore the 48-unit classifier, and replace the shared attention kernel with separately learned, identity-initialized average and maximum kernels.
mechanism: Descriptor-specific dual-statistic channel attention
evidence_used: The shared dual-statistic design achieved 9,320 correct, while reallocating classifier capacity to the extra spatial block fell to 9,261; separate three-weight kernels preserve the successful signals while allowing their distinct statistics to be interpreted independently.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249858, "training_seconds": 73.8674339579884, "validation_accuracy": 0.93, "validation_correct": 9300, "validation_cross_entropy": 0.19352327423095703, "validation_score": 9300.418927733372}

RECENT RESULT
hypothesis: Restoring the 9,320-correct shared dual-statistic attention model and adding mild per-image translations will exceed 9,320 correct predictions by improving spatial robustness without changing parameter count or the successful paired-view objective.
change: Restore Reference Design 1, then apply independent replicate-padded translations of up to two pixels before its random flip and paired-orientation training.
mechanism: Per-image integer-translation augmentation with paired flip supervision
evidence_used: Shared average-plus-maximum attention with paired supervision achieved the best result at 9,320 correct. Translations previously appeared only in a confounded 9,024-correct experiment that also changed batch size and classifier capacity, so isolating translation augmentation on the best regimen is informative.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 61.5085795421619, "validation_accuracy": 0.9178, "validation_correct": 9178, "validation_cross_entropy": 0.2245953540802002, "validation_score": 9178.408298135653}

RECENT RESULT
hypothesis: Allowing the 9,320-correct shared attention model to learn a bounded relative weight for global-maximum evidence will exceed 9,320 correct predictions while preserving its beneficial shared kernel and initial behavior.
change: Add one scalar parameter, initialized to reproduce the current average-plus-maximum gate exactly, that learns the maximum descriptor’s weight in the range zero to two.
mechanism: Bounded learned dual-statistic balancing
evidence_used: Shared average-plus-maximum attention achieved 9,320 correct, whereas fully independent descriptor kernels fell to 9,300; a single learned balance retains weight sharing while adding only one controlled degree of freedom.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249856, "training_seconds": 63.763402458047494, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.19342187652587892, "validation_score": 9310.41896332708}

RECENT RESULT
hypothesis: Replacing hard global maxima with the mean of each channel’s top four spatial activations while restoring the shared attention kernel will exceed 9,320 correct predictions by retaining salient evidence with less outlier sensitivity and denser gradients.
change: Restore the best shared channel-attention kernel and replace its global-maximum descriptor with a parameter-free top-four activation mean; preserve paired-view training and flip-ensemble evaluation.
mechanism: Shared average-plus-top-k channel attention
evidence_used: Shared average-plus-maximum attention reached 9,320 correct, outperforming average-only attention at 9,286; separate kernels fell to 9,300 and learned descriptor balancing reached 9,310, motivating preserved sharing with a more robust salient-activation statistic.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 78.7983277910389, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19255116729736327, "validation_score": 9322.41926922191}

RECENT RESULT
hypothesis: Extending the verified top-four saliency statistic to the spatial-attention summary will exceed 9,322 correct predictions by reducing sensitivity to single-channel activation outliers while preserving salient evidence.
change: Restore average-plus-top-four channel attention and replace the spatial gate’s hard channel maximum with the mean of its four strongest channel activations.
mechanism: Top-four robust channel-and-spatial attention
evidence_used: Average-plus-top-four channel attention achieved 9,322 correct versus 9,320 for hard maxima and 9,286 for average-only attention, motivating the same parameter-free robust saliency statistic in the spatial branch.
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
