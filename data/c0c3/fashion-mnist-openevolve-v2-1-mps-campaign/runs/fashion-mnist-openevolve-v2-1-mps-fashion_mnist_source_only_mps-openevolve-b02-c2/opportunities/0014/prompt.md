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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 81.16789445909671, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.19513330612182617, "validation_score": 9286.418363372051}
prior_hypothesis: Restoring the 9,271-correct attention model and jointly supervising every image in both horizontal orientations will exceed 9,271 correct predictions by aligning training with the successful flip-ensemble evaluation without adding parameters.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249855, "training_seconds": 75.28254275000654, "validation_accuracy": 0.932, "validation_correct": 9320, "validation_cross_entropy": 0.1922616943359375, "validation_score": 9320.4193710176}
prior_hypothesis: Restoring plain paired-view cross-entropy and augmenting the successful channel attention with shared global-max evidence will exceed 9,286 correct predictions without adding parameters.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249588, "training_seconds": 74.68625958287157, "validation_accuracy": 0.9261, "validation_correct": 9261, "validation_cross_entropy": 0.19783590965270997, "validation_score": 9261.417419444491}
prior_hypothesis: Restoring the 9,320-correct dual-statistic attention and paired-view supervision, then reallocating one classifier unit to a parameter-efficient 7×7 residual block, will exceed 9,320 correct predictions while remaining below 250,000 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: A 232,778-parameter two-block residual CNN trained with 64-image batches and mild translation augmentation will exceed the reference design’s 9,141 correct predictions by shifting capacity from its large dense layer into spatial feature extraction and providing roughly twice as many optimizer updates.
change: Add a second residual block, reduce the dense head width, use batch size 64, introduce random two-pixel translations, and retune AdamW’s learning rate for the smaller batch.
mechanism: Deeper residual feature learning with parameter-efficient head and translated crops
evidence_used: The 209,146-parameter residual reference reached 91.41% after 782 updates, substantially outperforming the shallow model; its 150,528-weight first dense layer leaves room to deepen convolutional processing while remaining below the parameter ceiling.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 232778, "training_seconds": 67.78072387492284, "validation_accuracy": 0.9024, "validation_correct": 9024, "validation_cross_entropy": 0.25752065048217776, "validation_score": 9024.397607784658}

RECENT RESULT
hypothesis: Retaining the verified 209,146-parameter, batch-128 residual design while averaging original and mirrored validation predictions will exceed 9,141 correct predictions because its training distribution already includes 50% horizontal flips.
change: Adopt Reference Design 1’s architecture and training regimen, then add parameter-free horizontal-flip probability averaging during evaluation.
mechanism: Training-aligned horizontal-flip test-time ensemble
evidence_used: Reference Design 1 achieved 9,141 correct predictions, while the deeper batch-64 design with translation fell to 9,024; this preserves the winning capacity and optimization choices and exploits its already-trained flip invariance without adding training noise.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 209146, "training_seconds": 21.856981082819402, "validation_accuracy": 0.9161, "validation_correct": 9161, "validation_cross_entropy": 0.23078228607177734, "validation_score": 9161.406245690776}

RECENT RESULT
hypothesis: Adding a parameter-efficient 32-channel residual block before the first pooling layer while preserving the verified batch-128 training and flip ensemble will exceed 9,161 correct predictions.
change: Add two normalized 3×3 convolutions in a residual block at 28×28 resolution, increasing capacity from 209,146 to approximately 227,706 parameters without altering the successful optimizer, augmentation, classifier, or evaluation ensemble.
mechanism: High-resolution residual feature refinement
evidence_used: The 209,146-parameter residual design reached 9,161 correct, while the 232,778-parameter experiment fell to 9,024 after simultaneously shrinking its classifier, changing batch size, and adding translations; this patch isolates additional spatial depth while retaining every verified winning training choice.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 227706, "training_seconds": 33.428059292025864, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.21726366081237794, "validation_score": 9214.410757353642}

RECENT RESULT
hypothesis: Extending the 9,214-correct Reference Design 3 with a 32-channel bottleneck residual block at 7×7 resolution will exceed 9,214 correct predictions while retaining its successful batch-128 training, classifier capacity, augmentation, and flip ensemble.
change: Adopt Reference Design 3’s high-resolution residual block and evaluation ensemble, then add a 64→32→32→64 bottleneck residual block before the classifier, bringing the model to approximately 241,274 learned parameters.
mechanism: Low-resolution bottleneck residual refinement
evidence_used: Reference Design 3 improved from 9,161 to 9,214 correct by adding residual spatial refinement while preserving the proven regimen; the bottleneck adds further spatial processing within the parameter ceiling without the classifier shrinkage, batch-size change, or translation augmentation that confounded Reference Design 2.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241274, "training_seconds": 41.21943400008604, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.2078806953430176, "validation_score": 9236.413948167172}

RECENT RESULT
hypothesis: Restoring the verified 9,236-correct design and adding a second 24-channel bottleneck residual block at 7×7 resolution will exceed 9,236 correct predictions while remaining under the 250,000-parameter ceiling.
change: Restore batch-128 flip-only training, the 48-unit classifier, flip-ensemble evaluation, and 2.5e-3 schedule from Reference Design 2, then add a second low-resolution bottleneck block for approximately 249,754 total parameters.
mechanism: Stacked low-resolution bottleneck residual refinement
evidence_used: Reference Design 2 achieved 9,236 correct after its added 32-channel low-resolution residual block improved Reference Design 3 by 22 correct; another narrower residual block isolates whether additional spatial depth continues that gain without changing the successful regimen.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249754, "training_seconds": 49.69561283313669, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21128562622070313, "validation_score": 9245.412784556489}

RECENT RESULT
hypothesis: Averaging the final half of the optimization trajectory with decay 0.98 will exceed 9,245 correct predictions by reducing endpoint variance without changing the architecture or exposure budget.
change: Preserve the verified training regimen while maintaining an EMA of learned parameters during the final 391 optimizer steps and installing the averaged parameters after the last step.
mechanism: Late-training exponential weight averaging
evidence_used: The current 249,754-parameter design gained only 9 correct predictions from its latest bottleneck and has slightly worse cross-entropy than the 241,274-parameter predecessor; with almost no parameter headroom remaining, stabilizing its late cosine-decayed trajectory is the most direct isolated opportunity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249754, "training_seconds": 56.74616395798512, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2112758945465088, "validation_score": 9240.412787872896}

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
