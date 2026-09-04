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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 70.92706137499772, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.21391976051330566, "validation_score": 9239.411888838344}
prior_hypothesis: Reducing batch size from 128 to 96 will exceed 9,204 correct predictions by increasing optimizer updates from 782 to about 1,042 while avoiding the runtime cost that caused batch size 64 to time out.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding the four missing diagonal one-pixel views will exceed 9,159 correct predictions by more completely averaging the local translations that improved the prior two-view ensemble by 25 images.
change: Expand validation inference from five spatial views and their flips to all nine offsets in the centered 3×3 translation neighborhood and their flips.
mechanism: Full 3×3 translation-and-flip test-time ensemble
evidence_used: The prior augmentation-matched cardinal-translation ensemble improved validation correctness from 9,134 to 9,159 without changing training; diagonal offsets complete the same one-pixel invariance neighborhood.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 52.5878084579017, "validation_accuracy": 0.9153, "validation_correct": 9153, "validation_cross_entropy": 0.23746159172058107, "validation_score": 9153.404052944628}

RECENT RESULT
hypothesis: Reducing the batch size from 128 to 64 will exceed 9,159 correct predictions by providing about 1,563 optimizer updates instead of 782 while preserving the successful architecture, augmentation, schedule, and cardinal-view ensemble.
change: Halve the training batch size without changing any other training or inference behavior.
mechanism: Smaller-batch optimization under fixed exposure
evidence_used: The earlier change that doubled optimizer updates from 392 to 782 was part of an improvement from 8,928 to 9,134 correct; isolating another update-frequency increase is the most informative next test after additional diagonal inference views failed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the final iterate with a bias-controlled EMA of the 782 training iterates will exceed 9,159 correct predictions by reducing late-stage optimization noise without the timeout incurred by batch size 64.
change: Add an AdamW subclass that tracks model parameters and floating-point buffers after every update, then installs the averaged state after the final optimizer step; retain batch size 128, training, and cardinal-view inference.
mechanism: Exponential moving average of weights and BatchNorm state
evidence_used: Cardinal translation ensembling improved correctness from 9,134 to 9,159, while additional diagonal views regressed and batch size 64 timed out; this motivates a low-overhead optimization improvement without expanding validation inference or optimizer-step count.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Arithmetic probability averaging across the existing ten views will exceed 9,159 correct predictions by reducing the disproportionate effect of a poorly aligned view on the true class.
change: Replace logit averaging with numerically stable averaging of per-view class probabilities while leaving training, views, parameters, and runtime unchanged.
mechanism: Probability-space cardinal-view ensembling
evidence_used: Cardinal views improved correctness from 9,134 to 9,159, while adding diagonal views reduced it to 9,153, indicating heterogeneous view reliability; probability averaging is less punitive than the current geometric pooling when one translation strongly disagrees.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 53.73887754185125, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.23889026565551758, "validation_score": 9155.403586995444}

RECENT RESULT
hypothesis: Doubling the centered views’ weight while retaining cardinal translations will exceed 9,159 correct predictions by preserving translation robustness while reducing errors from displaced views.
change: Give the original and horizontally flipped original images twice the logit weight of each translated view, with normalization adjusted accordingly.
mechanism: Center-weighted logit ensembling
evidence_used: Equal-weight cardinal views improved correctness from 9,134 to 9,159, but adding more displaced views regressed to 9,153 and probability pooling reached only 9,155; this motivates a conservative center-biased refinement of the successful logit ensemble.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 51.34214787511155, "validation_accuracy": 0.9162, "validation_correct": 9162, "validation_cross_entropy": 0.2387619441986084, "validation_score": 9162.403628802404}

RECENT RESULT
hypothesis: Increasing the centered views’ relative weight from two to three will exceed 9,162 correct predictions by retaining useful cardinal-view robustness while moving further toward the centered inference that the successful prior weighting change favored.
change: Give each centered view three times the logit weight of each translated view and update normalization from twelve to fourteen total view-weight units.
mechanism: Threefold center-weighted logit ensembling
evidence_used: Equal-weight cardinal ensembling reached 9,159 correct, while doubling centered-view weight improved this to 9,162; testing the next center-weight dose is the clearest continuation of that positive result.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 43.55700908321887, "validation_accuracy": 0.9157, "validation_correct": 9157, "validation_cross_entropy": 0.23912928085327148, "validation_score": 9157.403509147694}

RECENT RESULT
hypothesis: Restricting training translations from ±2 pixels to ±1 will exceed 9,162 correct predictions by matching the successful validation views and reducing exposure spent on strongly displaced images.
change: Change random training crops from the five-by-five ±2 translation grid to the three-by-three ±1 grid; retain flips, optimization, architecture, and center-weighted inference.
mechanism: Augmentation-matched one-pixel translation training
evidence_used: One-pixel cardinal inference improved correctness from 9,134 to 9,159, while adding more displaced inference influence regressed and center weighting improved the result to 9,162; this suggests moderate translation robustness helps but excessive displacement is harmful.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 42.766273458022624, "validation_accuracy": 0.9175, "validation_correct": 9175, "validation_cross_entropy": 0.22528525276184083, "validation_score": 9175.408068242781}

RECENT RESULT
hypothesis: Reallocating the location-sensitive flattened head into a 96-channel local/dilated stage and pooled coarse-shape/saliency head will exceed 9,175 correct predictions by learning more discriminative garment-scale structure.
change: Add a parallel standard/dilated residual block, widen the final representation from 64 to 96 channels, and replace the 7×7 flattening bottleneck with 2×2 average plus global-max pooling.
mechanism: Parallel local/dilated residual features with spatial-pyramid classification
evidence_used: Augmentation and inference refinements plateaued near 9,175 correct, while the current model spends 94,420 parameters compressing exact 7×7 positions into only 30 features. This patch instead uses 247,142 parameters to learn local and larger-context features while retaining coarse layout and translation-tolerant saliency.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sampling only centered and cardinal one-pixel translations, with the center sampled twice, will exceed 9,175 correct predictions by removing harmful diagonal exposure and matching the successful center-weighted validation ensemble.
change: Replace uniform sampling over all nine ±1 translations with a six-entry distribution containing two centered and four cardinal offsets.
mechanism: Inference-distribution-matched geometric augmentation
evidence_used: Restricting training from ±2 to ±1 improved correctness from 9,162 to 9,175; diagonal validation views regressed, while doubling the centered-view weight improved inference, motivating the same center/cardinal distribution during training.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 50.12612995808013, "validation_accuracy": 0.9204, "validation_correct": 9204, "validation_cross_entropy": 0.21821519927978517, "validation_score": 9204.41043651425}

RECENT RESULT
hypothesis: Adding lightweight channel-attention gates to every residual block will exceed 9,204 correct predictions by improving feature selection while preserving the verified architecture, augmentation, and optimization behavior.
change: Add identity-initialized squeeze-and-excitation gates to all residual blocks, increasing learned parameters from 245,044 to 247,528 without changing exposure, batch size, or inference views.
mechanism: Identity-centered squeeze-and-excitation residual recalibration
evidence_used: The current topology reaches 9,204 correct and leaves 4,956 parameters unused, while the prior large architectural redesign could not be verified; a 2,484-parameter, topology-preserving attention addition is a lower-risk test of additional representational capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing batch size from 128 to 96 will exceed 9,204 correct predictions by increasing optimizer updates from 782 to about 1,042 while avoiding the runtime cost that caused batch size 64 to time out.
change: Change only the training batch size from 128 to 96, preserving the verified architecture, matched center/cardinal augmentation, schedule, and inference ensemble.
mechanism: Intermediate smaller-batch optimization
evidence_used: The 9,204-correct design is the strongest verified implementation; batch size 64 was motivated by additional optimizer updates but timed out, so 96 is the most informative intermediate runtime–update tradeoff.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 70.92706137499772, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.21391976051330566, "validation_score": 9239.411888838344}

RECENT RESULT
hypothesis: Reducing batch size from 96 to 88 will exceed 9,239 correct predictions by increasing optimizer steps from 1,042 to about 1,137 while remaining substantially closer to the verified runtime than the timed-out batch-size-64 design.
change: Change only the training batch size from 96 to 88, preserving the architecture, optimizer, schedule, matched center/cardinal augmentation, and inference ensemble.
mechanism: Moderately smaller-batch optimization
evidence_used: Reducing batch size from 128 to 96 improved validation correctness from 9,204 to 9,239, whereas 64 timed out; 88 is a conservative continuation of the demonstrated update-frequency benefit.
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
