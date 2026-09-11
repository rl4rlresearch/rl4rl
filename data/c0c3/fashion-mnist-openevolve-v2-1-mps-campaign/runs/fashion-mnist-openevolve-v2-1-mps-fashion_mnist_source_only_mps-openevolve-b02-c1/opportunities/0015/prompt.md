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
verified_results: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.8552653749939, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.1980760238647461, "validation_score": 9280.417335786746}
prior_hypothesis: A batch size of 48 will finish within the verification limit and exceed 9,238 correct predictions by providing 33% more optimizer updates than batch size 64 without the roughly doubled runtime that caused batch size 32 to time out.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging centered, one-pixel translated, and mirrored predictions will exceed 9,129 correct validations because translation augmentation already trains the classifier across shifted views, paralleling the successful 32-correct flip ensemble gain.
change: Extend evaluation-time augmentation from two mirrored views to ten views spanning the centered image and four one-pixel cardinal translations, averaging their class probabilities.
mechanism: Flip-and-translation probability ensemble
evidence_used: Flip probability averaging improved the same trained model from 9,097 to 9,129 correct while lowering cross-entropy; training also applies random ±2-pixel translations, making translation ensembling the closest untested analogue.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 29.67150841699913, "validation_accuracy": 0.9142, "validation_correct": 9142, "validation_cross_entropy": 0.23158731079101563, "validation_score": 9142.405980149048}

RECENT RESULT
hypothesis: Averaging all nine one-pixel translation views and their mirrors will exceed 9,142 correct predictions because the five-view translation ensemble already improved the flip ensemble by 13 correct predictions.
change: Add the four missing diagonal one-pixel validation views and normalize the resulting 18-view probability ensemble.
mechanism: Full 3×3 translation-and-flip probability ensemble
evidence_used: Cardinal translation-and-flip averaging improved validation_correct from 9,129 to 9,142 while lowering cross-entropy, and the unchanged training procedure exposes the model to diagonal translations as part of its random ±2-pixel crops.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 29.758014041930437, "validation_accuracy": 0.9137, "validation_correct": 9137, "validation_cross_entropy": 0.23122888870239258, "validation_score": 9137.406098333615}

RECENT RESULT
hypothesis: Giving the centered original and mirrored views twice the weight of each translated view will exceed 9,142 correct predictions by retaining the cardinal ensemble’s gain while reducing translation-induced errors evidenced by the worse full-grid ensemble.
change: Weight both centered predictions by 2× and normalize the resulting probability ensemble by its total weight of 12.
mechanism: Center-weighted cardinal translation ensemble
evidence_used: Equal-weight cardinal translation ensembling improved validation_correct from 9,129 to 9,142, whereas adding four more translated views reduced it to 9,137; this suggests translated predictions help but should receive less influence than centered views.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 30.62616812484339, "validation_accuracy": 0.9148, "validation_correct": 9148, "validation_cross_entropy": 0.23178596572875976, "validation_score": 9148.405914675042}

RECENT RESULT
hypothesis: Weighting each centered prediction 3× will exceed 9,148 correct by further reducing translation-induced errors while retaining useful translated-view evidence.
change: Increase the original and mirrored centered-view weights from 2× to 3× and normalize the ensemble’s total weight from 12 to 14.
mechanism: More strongly center-weighted cardinal translation ensemble
evidence_used: Increasing centered-view weight from 1× to 2× improved validation_correct from 9,142 to 9,148, while the translation-free flip ensemble reached only 9,129; this supports testing the next intermediate reduction in translated-view influence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 25.15302133304067, "validation_accuracy": 0.9147, "validation_correct": 9147, "validation_cross_entropy": 0.23199610557556152, "validation_score": 9147.405845438745}

RECENT RESULT
hypothesis: Sampling centered images with probability 1/3 and each one-pixel cardinal translation with probability 1/6 will exceed 9,148 correct predictions by matching training augmentation to the empirically strongest center-weighted validation ensemble.
change: Replace uniform ±2-pixel crop augmentation with a center-weighted mixture of centered and one-pixel cardinal views; retain random horizontal flips and all other settings.
mechanism: Train–test geometric distribution alignment
evidence_used: The center-weighted cardinal ensemble reached 9,148 correct, outperforming equal weighting at 9,142 and the full translation grid at 9,137, indicating that centered and cardinal views are useful while diagonal or excessive translation influence is harmful.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232682, "training_seconds": 37.01068200008012, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.20636922912597655, "validation_score": 9232.414466804961}

RECENT RESULT
hypothesis: Reducing the batch size from 128 to 64 will exceed 9,232 correct predictions by doubling optimizer updates under the fixed 100,000-example exposure while retaining sufficiently stable batch-normalization statistics.
change: Set the training batch size to 64 while preserving the architecture, augmentation, optimizer, and exposure-relative learning-rate schedule.
mechanism: Higher-update stochastic optimization
evidence_used: The earlier move from 392 to 782 optimizer steps accompanied an improvement from 8,928 to 9,097 correct predictions; the current augmentation-aligned design now reaches 9,232 correct but still performs only 782 updates, motivating a controlled test of additional optimization steps.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 232682, "training_seconds": 64.92380745802075, "validation_accuracy": 0.9238, "validation_correct": 9238, "validation_cross_entropy": 0.20280714263916017, "validation_score": 9238.415694239147}

RECENT RESULT
hypothesis: Reducing the batch size from 64 to 32 will exceed 9,238 correct predictions by doubling optimizer updates again while retaining adequate batch-normalization statistics.
change: Set the training batch size to 32 while preserving the architecture, augmentation, optimizer, and exposure-relative learning-rate schedule.
mechanism: Higher-update stochastic optimization
evidence_used: The controlled reduction from batch size 128 to 64 doubled optimizer steps from 782 to 1,564 and improved validation_correct from 9,232 to 9,238 while lowering cross-entropy from 0.20637 to 0.20281, directly motivating another batch-size reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing destructive max pooling and the large position-specific flattening head with residual learned downsampling and global mean–max feature aggregation will exceed 9,238 correct validation predictions.
change: Replace the sequential CNN with a seven-convolution residual pyramid that preserves information through projection shortcuts and classifies pooled channel statistics; retain the proven augmentation, optimizer, batch size, schedule, and test-time ensemble.
mechanism: Residual learned-downsampling pyramid with global mean–max pooling
evidence_used: The current design reached 9,238 correct, while another batch-size reduction timed out, limiting further gains from additional optimizer steps. Roughly half of its parameters are committed to the 3×3 flattening head, so reallocating capacity to residual spatial feature extraction tests a genuinely different representation within the parameter and runtime constraints.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A batch size of 48 will finish within the verification limit and exceed 9,238 correct predictions by providing 33% more optimizer updates than batch size 64 without the roughly doubled runtime that caused batch size 32 to time out.
change: Reduce the training batch size from 64 to 48 while preserving the proven architecture, augmentation, optimizer, schedule, and validation ensemble.
mechanism: Intermediate-batch higher-update optimization
evidence_used: Reducing batch size from 128 to 64 doubled updates and improved validation_correct from 9,232 to 9,238 while lowering cross-entropy; batch size 32 timed out, motivating an intermediate setting.
result: improved the objective and became an available design
reported_values: {"batch_size": 48, "examples_processed": 100000, "optimizer_steps": 2084, "parameters": 232682, "training_seconds": 73.8552653749939, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.1980760238647461, "validation_score": 9280.417335786746}

RECENT RESULT
hypothesis: Reducing batch size from 48 to 40 will exceed 9,280 correct predictions by increasing optimizer steps from 2,084 to approximately 2,500 while remaining faster than the batch-32 configuration that timed out.
change: Change only the training batch size from 48 to 40, preserving the proven architecture, augmentation, optimizer, schedule, and validation ensemble.
mechanism: Intermediate-batch higher-update optimization
evidence_used: Reducing batch size from 64 to 48 increased optimizer steps by 33% and improved validation_correct from 9,238 to 9,280; batch size 32 timed out, so 40 is the next conservative point between the best completed run and the runtime boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging learned parameters over the latter half of training will exceed 9,280 correct predictions by reducing variance from batch-48 optimization without requiring the slower batch-40 update count.
change: Maintain a low-overhead EMA every two optimizer steps during the second half of training, then install the averaged parameters after the final step.
mechanism: Tail exponential parameter averaging
evidence_used: Batch 48 improved validation_correct from 9,238 to 9,280, while batch 40 timed out; parameter averaging tests whether the benefits of smoother optimization can be captured within the proven runtime regime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Expanding the final convolutional stage from 96 to 106 channels will exceed 9,280 correct predictions by increasing learned feature capacity with minimal additional runtime.
change: Widen the final feature stage and matching classifier input, increasing parameters from 232,682 to 249,982 while preserving every proven training and evaluation setting.
mechanism: Near-ceiling final-stage channel expansion
evidence_used: The batch-48 design completed in 73.86 seconds and achieved 9,280 correct, while slower optimization and residual redesigns timed out; this controlled expansion uses nearly all remaining parameter headroom with only about 1–2% additional convolutional work.
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
