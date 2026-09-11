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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 42.766273458022624, "validation_accuracy": 0.9175, "validation_correct": 9175, "validation_cross_entropy": 0.22528525276184083, "validation_score": 9175.408068242781}
prior_hypothesis: Restricting training translations from ±2 pixels to ±1 will exceed 9,162 correct predictions by matching the successful validation views and reducing exposure spent on strongly displaced images.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the 105,866-parameter baseline to a batch-normalized 245,044-parameter residual network, while doubling optimizer updates and adding Fashion-MNIST-compatible augmentation, will exceed 8,928 correct validation predictions within the same 100,000-example budget.
change: Use a deeper residual model, batch size 128, random translations and horizontal flips, unsmoothed cross-entropy, short warmup with cosine decay, and horizontal-flip test-time ensembling.
mechanism: Residual CNN with mild geometric augmentation and flip-ensemble inference
evidence_used: The starting model reaches 8,928 correct with only two convolutional layers and 105,866 parameters, leaving substantial capacity under the 250,000-parameter ceiling and only 392 optimizer steps.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 39.47495845798403, "validation_accuracy": 0.9134, "validation_correct": 9134, "validation_cross_entropy": 0.2425220588684082, "validation_score": 9134.402407342735}

RECENT RESULT
hypothesis: Averaging predictions across the centered image and four one-pixel translations, with their horizontal flips, will exceed 9,134 correct predictions by converting training-time translation robustness into more stable validation decisions.
change: Replace two-view flip inference with a ten-view sequential ensemble over the original image, four cardinal translations, and horizontal flips, while leaving training unchanged.
mechanism: Augmentation-matched translation-and-flip test-time ensemble
evidence_used: The 9,134-correct design trains on random translations up to two pixels but ensembles only horizontal flips at validation; symmetric one-pixel translation views directly exploit that learned invariance without changing exposure, optimization, or parameter count.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 39.26220383308828, "validation_accuracy": 0.9159, "validation_correct": 9159, "validation_cross_entropy": 0.23834408340454102, "validation_score": 9159.40376500094}

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
