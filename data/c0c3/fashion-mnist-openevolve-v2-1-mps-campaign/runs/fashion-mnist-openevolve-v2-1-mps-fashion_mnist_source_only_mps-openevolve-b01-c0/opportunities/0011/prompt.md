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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216346, "training_seconds": 66.88195800012909, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.22626232528686524, "validation_score": 9265.407743098429}
prior_hypothesis: Sampling only the centered and four one-pixel cardinal translations during training will exceed 9,188 correct predictions by concentrating the fixed exposure on the spatial views that improved evaluation performance.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing useful capacity to 216,346 parameters, doubling optimizer updates, and training translation/flip invariance will exceed the current 8,928 correct predictions after the same 100,000-example exposure.
change: Replace the shallow CNN with a four-convolution network, use batch size 128, per-image random translations and flips, evaluation-time flip averaging, decoupled AdamW parameter groups, and a warmup-cosine schedule.
mechanism: Batch-normalized deeper CNN with per-image geometric augmentation and flip-ensemble inference
evidence_used: The starting 105,866-parameter two-layer CNN reaches 8,928 correct with only 392 optimizer steps; substantial parameter headroom and the short fixed exposure motivate a higher-capacity normalized model with more update opportunities.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 34.70552358403802, "validation_accuracy": 0.9138, "validation_correct": 9138, "validation_cross_entropy": 0.2494143997192383, "validation_score": 9138.40018747992}

RECENT RESULT
hypothesis: Averaging predictions across the centered image and four one-pixel translations, each with its horizontal flip, will exceed 9,138 correct predictions because the verified model was explicitly trained for translation and flip invariance.
change: Replace two-view logit averaging during evaluation with ten-view arithmetic probability averaging over centered, translated, and flipped inputs; training remains unchanged.
mechanism: Translation-flip probability ensemble
evidence_used: The current 216,346-parameter model reached 9,138 correct while training on random translations and flips, but evaluation exploits only flip invariance; translation ensembling directly uses the already-learned augmentation invariance without changing the fixed training exposure.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 28.460448458092287, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.25047449645996095, "validation_score": 9167.399848218749}

RECENT RESULT
hypothesis: Adding the four diagonal one-pixel translations to evaluation-time probability averaging will exceed 9,167 correct predictions by more completely exploiting the model’s learned translation and flip invariance.
change: Expand evaluation from five spatial views and their flips to all nine positions in the centered 3×3 translation grid and their flips.
mechanism: Full 3×3 translation-flip probability ensemble
evidence_used: Moving from flip-only evaluation to centered-plus-cardinal translation/flip averaging increased validation correct from 9,138 to 9,167 without retraining, showing that translation ensembling is beneficial; diagonal views are the closest untested extension of that successful mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 33.043977125082165, "validation_accuracy": 0.9166, "validation_correct": 9166, "validation_cross_entropy": 0.25138679122924806, "validation_score": 9166.399556718598}

RECENT RESULT
hypothesis: Ensembling the final model with a short-horizon EMA of its second-half trajectory will exceed 9,167 correct predictions by reducing augmentation and optimization variance.
change: Track an EMA of parameters and floating-point BatchNorm buffers during the second half of training, then average its ten-view probabilities with the final model’s ten-view probabilities.
mechanism: Late-training exponential weight averaging with prediction ensembling
evidence_used: Translation-flip probability ensembling improved validation correct from 9,138 to 9,167, showing that averaging complementary predictions benefits this model; temporal weight ensembling extends that successful variance-reduction mechanism without changing training exposure or learned-parameter count.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 26.90609429194592, "validation_accuracy": 0.9177, "validation_correct": 9177, "validation_cross_entropy": 0.24972009811401366, "validation_score": 9177.400089588664}

RECENT RESULT
hypothesis: Averaging the final model with both short-horizon (0.98) and long-horizon (0.995) late-training EMAs will exceed 9,177 correct predictions by combining complementary levels of optimization-variance reduction.
change: Track two second-half EMA states and average both of their ten-view probability predictions with the final model.
mechanism: Multi-timescale temporal weight ensembling
evidence_used: Adding one 0.98 EMA ensemble increased validation correct from 9,167 to 9,177; this directly supports extending temporal ensembling with a longer-timescale average while preserving the successful final and short-EMA predictors.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 40.319087208015844, "validation_accuracy": 0.916, "validation_correct": 9160, "validation_cross_entropy": 0.2515191177368164, "validation_score": 9160.399514472383}

RECENT RESULT
hypothesis: Replacing dense-head capacity with a fifth convolution at 7×7 resolution will exceed 9,177 correct predictions by learning stronger spatial features while retaining nearly the same parameter count.
change: Add a post-pooling 64-channel convolution and narrow the dense bottleneck from 48 to 36 units, reducing total learned parameters from 216,346 to 215,550.
mechanism: Convolutional capacity reallocation
evidence_used: The four-convolution redesign improved the objective from 8,928 to 9,138 correct, while the longer-timescale EMA extension regressed to 9,160; this favors testing additional representational depth instead of further temporal ensembling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 215550, "training_seconds": 33.078072499949485, "validation_accuracy": 0.9157, "validation_correct": 9157, "validation_cross_entropy": 0.2438549213409424, "validation_score": 9157.40197614}

RECENT RESULT
hypothesis: Doubling optimizer steps from 782 to about 1,563 while applying square-root learning-rate scaling will exceed 9,177 correct predictions by improving optimization under the fixed exposure without destabilizing updates.
change: Reduce batch size from 128 to 64, scale learning rates by approximately √½, and increase EMA decay from 0.98 to 0.99 so its averaging horizon remains approximately constant in examples.
mechanism: Smaller-batch optimization with exposure-matched EMA
evidence_used: The largest prior gain coincided with doubling optimizer updates from 392 to 782, whereas additional EMA timescales and convolutional reallocation regressed; another controlled increase in update opportunities is therefore the most informative remaining lever.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216346, "training_seconds": 60.05283995810896, "validation_accuracy": 0.9188, "validation_correct": 9188, "validation_cross_entropy": 0.2411598472595215, "validation_score": 9188.402848997333}

RECENT RESULT
hypothesis: Reducing batch size from 64 to 32 will exceed 9,188 correct predictions by doubling optimizer updates again, while square-root learning-rate scaling and an EMA decay of 0.995 preserve update stability and approximately the same averaging horizon in examples.
change: Use batch size 32, scale all learning-rate schedule values by approximately √½, and increase EMA decay from 0.99 to 0.995.
mechanism: Further smaller-batch optimization with exposure-matched learning rate and EMA
evidence_used: The previous controlled reduction from batch size 128 to 64 improved validation correct from 9,177 to 9,188; this directly motivates testing the same exposure-matched step-doubling mechanism once more.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch size 48 will exceed 9,188 correct predictions by providing about 2,084 optimizer updates while retaining sufficient throughput to finish verification; exposure-matched learning rates and EMA decay will preserve the stability of the successful batch-64 design.
change: Reduce batch size from 64 to 48, scale learning rates by √(48/64), and increase EMA decay to 0.9925 to maintain approximately the same averaging horizon in examples.
mechanism: Intermediate smaller-batch optimization with exposure-matched learning rate and EMA
evidence_used: Reducing batch size from 128 to 64 improved validation correct from 9,177 to 9,188, but batch size 32 timed out; batch size 48 tests an intermediate increase in update count with substantially less runtime risk.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 216346, "training_seconds": 58.067536499816924, "validation_accuracy": 0.9187, "validation_correct": 9187, "validation_cross_entropy": 0.24077623138427734, "validation_score": 9187.40297354781}

RECENT RESULT
hypothesis: Sampling only the centered and four one-pixel cardinal translations during training will exceed 9,188 correct predictions by concentrating the fixed exposure on the spatial views that improved evaluation performance.
change: Replace uniform sampling across 25 translations up to two pixels with uniform sampling across the five centered/cardinal one-pixel views used by evaluation; retain horizontal flips and all other training settings.
mechanism: Train–evaluation matched cardinal-translation augmentation
evidence_used: Cardinal translation-flip averaging improved validation correct from 9,138 to 9,167, while adding diagonal views reduced it to 9,166; the batch-size-48 experiment also failed to improve 9,188, motivating a train–evaluation augmentation alignment rather than another step-count change.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216346, "training_seconds": 66.88195800012909, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.22626232528686524, "validation_score": 9265.407743098429}



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
