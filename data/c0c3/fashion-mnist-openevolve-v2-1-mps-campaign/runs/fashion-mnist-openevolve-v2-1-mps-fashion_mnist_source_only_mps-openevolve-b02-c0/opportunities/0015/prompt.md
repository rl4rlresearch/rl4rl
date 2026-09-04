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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 75.64941224991344, "validation_accuracy": 0.9221, "validation_correct": 9221, "validation_cross_entropy": 0.24149708404541015, "validation_score": 9221.402739568563}
prior_hypothesis: Biasing one-pixel crops toward the centered position will exceed 9,203 correct predictions by further reducing training–validation translation mismatch while retaining useful local translation robustness.

## Recent verification evidence

RECENT RESULT
hypothesis: Keeping the proven 216,346-parameter architecture and training recipe but doubling optimizer updates from roughly 782 to 1,563 will exceed 9,128 correct predictions by improving optimization under the fixed 100,000-example exposure.
change: Reduce batch size from 128 to 64 while preserving the successful architecture, augmentation, optimizer, schedule, and flip-averaged inference.
mechanism: Small-batch optimizer-step densification
evidence_used: The successful design combined an increase from 392 to 782 optimizer steps with a gain from 8,928 to 9,128 correct, whereas replacing its spatial head with a deeper residual global-pooling model fell to 8,985; this motivates retaining the proven model and testing increased update density.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216346, "training_seconds": 46.52469175006263, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.2520628761291504, "validation_score": 9141.39934096724}

RECENT RESULT
hypothesis: Reducing batch size from 64 to 32 while preserving the proven architecture and recipe will exceed 9,141 correct predictions by providing roughly twice as many parameter updates within the fixed 100,000-example exposure.
change: Change only the training batch size from 64 to 32.
mechanism: Further small-batch optimizer-step densification
evidence_used: Reducing batch size from 128 to 64 increased optimizer steps from 782 to 1,564 and improved validation_correct from 9,128 to 9,141, while the architecture-changing residual alternative fell to 8,985; this supports another controlled update-density test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch size 48 will exceed 9,141 correct predictions by increasing optimizer updates beyond batch size 64 while avoiding the verification timeout observed at batch size 32.
change: Change only the training batch size from 64 to 48, preserving the proven architecture, augmentation, optimizer, and schedule.
mechanism: Intermediate small-batch optimizer-step densification
evidence_used: Reducing batch size from 128 to 64 increased optimizer steps from 782 to 1,564 and improved validation_correct from 9,128 to 9,141, while batch size 32 did not finish; batch size 48 tests the supported update-density trend within a safer runtime envelope.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 53.00537812500261, "validation_accuracy": 0.9146, "validation_correct": 9146, "validation_cross_entropy": 0.2526547607421875, "validation_score": 9146.399152276965}

RECENT RESULT
hypothesis: Reducing batch size from 48 to 44 will exceed 9,146 correct predictions by providing about 9% more optimizer updates while remaining within the runtime limit that batch size 32 exceeded.
change: Change only the training batch size from 48 to 44, preserving the best-performing architecture, augmentation, optimizer, and schedule.
mechanism: Conservative optimizer-step densification
evidence_used: Reducing batch size from 64 to 48 increased optimizer steps from 1,564 to 2,084 and validation_correct from 9,141 to 9,146, while batch size 32 timed out; 44 is the nearest conservative test of the demonstrated update-density trend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 44, "examples_processed": 100000, "optimizer_steps": 2274, "parameters": 216346, "training_seconds": 59.67123183305375, "validation_accuracy": 0.9142, "validation_correct": 9142, "validation_cross_entropy": 0.251830224609375, "validation_score": 9142.3994151844}

RECENT RESULT
hypothesis: Widening the proven spatial classifier head from 48 to 56 units will exceed 9,146 correct predictions while remaining below the 250,000-parameter ceiling.
change: Increase only the classifier bottleneck width, preserving batch size 48, convolutional features, augmentation, optimizer, schedule, and inference averaging.
mechanism: Dense bottleneck capacity expansion
evidence_used: The 216,346-parameter spatial-head model reached 9,146 correct, whereas replacing that head with a 239,562-parameter global-average residual model fell to 8,985; this motivates adding capacity to the successful head rather than reallocating it.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 241538, "training_seconds": 65.3889667079784, "validation_accuracy": 0.9138, "validation_correct": 9138, "validation_cross_entropy": 0.2520160591125488, "validation_score": 9138.39935589992}

RECENT RESULT
hypothesis: Averaging the successful batch-48 model’s parameter trajectory will exceed 9,146 correct predictions by reducing late-training optimizer noise without changing its proven architecture, exposure, or update count.
change: Maintain a bias-controlled exponential moving average of trainable parameters after every optimizer step and install the averaged weights after the final step.
mechanism: End-of-training exponential weight averaging
evidence_used: Batch size 48 achieved the best result at 9,146 correct, while further update densification at batch size 44 fell to 9,142 and widening the classifier fell to 9,138; this motivates preserving the best configuration and stabilizing its learned endpoint.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 76.66495912498794, "validation_accuracy": 0.9134, "validation_correct": 9134, "validation_cross_entropy": 0.25030693588256836, "validation_score": 9134.399901804629}

RECENT RESULT
hypothesis: Averaging predictions over the centered image and four one-pixel translations will exceed 9,146 correct by exploiting the translation invariance learned from random-crop augmentation without changing the proven training configuration.
change: Extend evaluation-time augmentation from two horizontal-flip views to ten views spanning center, one-pixel cardinal translations, and their flips.
mechanism: Translation-and-flip probability ensemble
evidence_used: The batch-48 architecture and training recipe remains best at 9,146 correct, while changing capacity, update density, and parameter averaging all regressed; retaining its learned solution and extending its already-successful flip averaging is the most targeted next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 79.37253633304499, "validation_accuracy": 0.9156, "validation_correct": 9156, "validation_cross_entropy": 0.2526120048522949, "validation_score": 9156.399165901383}

RECENT RESULT
hypothesis: Adding the four one-pixel diagonal translations to the successful cardinal-translation ensemble will exceed 9,156 correct predictions by more completely averaging the translation variability learned from random-crop augmentation.
change: Expand evaluation-time augmentation from five spatial views to the full 3×3 grid of one-pixel translations, retaining horizontal-flip averaging for every view.
mechanism: Diagonal translation-and-flip probability ensemble
evidence_used: Adding four one-pixel cardinal translations to flip-averaged inference improved validation_correct from 9,146 to 9,156 without changing training, directly motivating a controlled extension to the four omitted diagonal translations.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 75.77342379209585, "validation_accuracy": 0.9157, "validation_correct": 9157, "validation_cross_entropy": 0.2537933609008789, "validation_score": 9157.3987897971}

RECENT RESULT
hypothesis: Adding the four two-pixel cardinal translations to the successful 3×3 one-pixel ensemble will exceed 9,157 correct predictions by covering the full translation range used during padding-2 random-crop training.
change: Expand evaluation from nine spatial views to thirteen by adding centered two-pixel shifts in each cardinal direction, retaining horizontal-flip probability averaging.
mechanism: Training-range translation ensemble
evidence_used: One-pixel cardinal translation averaging improved validation_correct from 9,146 to 9,156, and adding diagonals reached 9,157; because training uses random offsets spanning two pixels from center, the untested two-pixel cardinal views are the most directly motivated extension.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Vectorizing the 13 spatial views and their flips into one inference batch will finish within the time limit and exceed 9,157 correct predictions by adding the four two-pixel cardinal translations without sequential-forward overhead.
change: Expand inference to the 3×3 one-pixel grid plus four two-pixel cardinal shifts, and evaluate all 26 translated/flipped views in one batched forward pass.
mechanism: Vectorized training-range translation-and-flip ensemble
evidence_used: The 3×3 ensemble improved validation_correct to 9,157, while the sequential 13-view extension timed out; batching its identical view set directly addresses that failure while testing the evidence-motivated full augmentation range.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 59.00248045893386, "validation_accuracy": 0.9149, "validation_correct": 9149, "validation_cross_entropy": 0.2579831604003906, "validation_score": 9149.397461600234}

RECENT RESULT
hypothesis: Restricting training translations from two pixels to one will exceed 9,157 correct predictions by concentrating the fixed exposure budget on the translation range that improved validation inference.
change: Change random-crop augmentation from a five-by-five offset grid to a three-by-three offset grid while preserving the model, optimizer, batch size, and best inference ensemble.
mechanism: One-pixel-aligned translation augmentation
evidence_used: One-pixel translation averaging improved validation_correct from 9,146 to 9,157, whereas adding two-pixel cardinal views reduced it to 9,149; this indicates that two-pixel shifts are less useful for the fixed validation distribution.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 64.54954916704446, "validation_accuracy": 0.9203, "validation_correct": 9203, "validation_cross_entropy": 0.24229328918457033, "validation_score": 9203.402481446494}

RECENT RESULT
hypothesis: Biasing one-pixel crops toward the centered position will exceed 9,203 correct predictions by further reducing training–validation translation mismatch while retaining useful local translation robustness.
change: Replace uniform per-axis crop offsets with a symmetric 1:2:1 distribution, increasing the centered crop probability from 1/9 to 1/4 without changing the translation range.
mechanism: Center-weighted one-pixel translation augmentation
evidence_used: Reducing augmentation from uniform two-pixel shifts to uniform one-pixel shifts improved validation_correct from 9,157 to 9,203, strongly motivating a further controlled reduction in average translation magnitude.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 75.64941224991344, "validation_accuracy": 0.9221, "validation_correct": 9221, "validation_cross_entropy": 0.24149708404541015, "validation_score": 9221.402739568563}



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
