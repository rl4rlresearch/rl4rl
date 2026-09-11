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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 63.53062425018288, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.21036614379882812, "validation_score": 9286.413098137751}
prior_hypothesis: Increasing the EMA interpolation rate from 0.01 to 0.02 will exceed 9,282 correct predictions by reducing parameter lag while retaining useful smoothing over roughly the final 50 optimizer steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing the EMA interpolation rate from 0.01 to 0.02 will exceed 9,282 correct predictions by reducing parameter lag while retaining useful smoothing over roughly the final 50 optimizer steps.
change: Shorten the learned-parameter EMA horizon from approximately 100 steps to 50 steps, leaving the architecture, paired-view objective, BatchNorm buffers, and schedule unchanged.
mechanism: Short-horizon parameter-only EMA
evidence_used: Parameter-only EMA matched the best 9,282 correct while lowering cross-entropy, whereas lengthening the EMA horizon from 0.01 to 0.005 previously reduced correct predictions from 9,237 to 9,233; this directly motivates testing less trajectory lag.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 63.53062425018288, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.21036614379882812, "validation_score": 9286.413098137751}

RECENT RESULT
hypothesis: Batch size 48 with square-root-scaled learning rate will exceed 9,286 correct predictions by providing 33% more optimizer updates than batch 64 while avoiding the timeout observed at batch 32.
change: Reduce batch size from 64 to 48 and scale the AdamW base and scheduled learning rates from 2.1e-3 to 1.8e-3.
mechanism: Intermediate small-batch paired-view optimization
evidence_used: Reducing batch size from 128 to 64 improved correct predictions from 9,212 to 9,215, while batch 32 timed out; batch 48 tests an intermediate update-frequency increase on the stronger paired-view, short-horizon EMA design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the EMA interpolation rate from 0.02 to 0.04 will exceed 9,286 correct predictions by further reducing end-of-training parameter lag while retaining smoothing over roughly 25 optimizer steps.
change: Shorten the learned-parameter EMA horizon from approximately 50 steps to 25 steps, leaving the proven architecture, paired-view objective, BatchNorm handling, and learning-rate schedule unchanged.
mechanism: Very-short-horizon parameter-only EMA
evidence_used: Raising the parameter-only EMA rate from 0.01 to 0.02 improved validation correct from 9,282 to 9,286, while the earlier longer-horizon 0.005 EMA regressed; another geometric reduction directly tests the observed benefit of lower EMA lag.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 75.16009229212068, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.21150547676086426, "validation_score": 9285.412709648937}

RECENT RESULT
hypothesis: An EMA interpolation rate of 0.03 will exceed 9,286 correct predictions by balancing the excess parameter lag at 0.01 against the reduced smoothing at 0.04.
change: Increase the learned-parameter EMA rate from 0.02 to 0.03, shortening its effective horizon from roughly 50 to 33 optimizer steps.
mechanism: Intermediate-horizon parameter-only EMA
evidence_used: Increasing the rate from 0.01 to 0.02 improved validation correct from 9,282 to 9,286, while increasing it further to 0.04 slightly regressed to 9,285; the untested midpoint directly probes the apparent optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 60.133658915990964, "validation_accuracy": 0.9286, "validation_correct": 9286, "validation_cross_entropy": 0.2109988639831543, "validation_score": 9286.41288230309}

RECENT RESULT
hypothesis: Reducing the final learning-rate floor from 10% to 1% will exceed 9,286 correct predictions by stabilizing the late trajectory averaged by the proven 0.02 parameter-only EMA.
change: Preserve the architecture, paired-view objective, optimizer, and EMA while changing cosine decay from a 2.1e-4 minimum learning rate to 2.1e-5.
mechanism: Low-floor cosine tail convergence
evidence_used: Shortening EMA lag from rate 0.01 to 0.02 improved correct predictions from 9,282 to 9,286, but further shortening to 0.04 regressed; reducing late parameter motion tests whether the successful 0.02 EMA can average a more converged trajectory without sacrificing smoothing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Arithmetic averaging of view probabilities will exceed 9,286 correct predictions by allowing either orientation to carry uncertain examples while preserving direct optimization of the mirrored ensemble.
change: Replace geometric probability averaging induced by mean logits with a normalized arithmetic probability mixture during both training and validation.
mechanism: Probability-space mirrored-view ensemble
evidence_used: Paired-view training improved correct predictions from 9,237 to 9,282, while auxiliary per-view supervision regressed to 9,252; this suggests the gain comes from complementary ensemble behavior rather than forcing each view to succeed independently.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 79.98436108394526, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20935391540527343, "validation_score": 9280.413443900607}

RECENT RESULT
hypothesis: Replacing the shared flattened MLP with ten class-specific attention queries over low-resolution image tokens will exceed 9,286 correct predictions by letting each class aggregate different long-range spatial evidence.
change: Retain the proven convolutional front end, mirrored-pair objective, optimizer, and EMA, but replace the flattened classifier with positional tokens, two lightweight global-attention layers, and one learned query/readout per class.
mechanism: Class-query global-attention readout
evidence_used: Deepening the existing max-pooled convolutional branches regressed from 9,212 to 9,196, while learned convolutional downsampling timed out. The 9,286-best design still assumes local features followed by a shared flattened representation; low-resolution class-query attention tests a different prediction mechanism without expensive high-resolution convolutions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Widening the classifier hidden layer from 48 to 58 units will exceed 9,286 correct predictions by using the remaining parameter budget to preserve more spatial-feature information without adding expensive convolutional computation.
change: Expand both classifier linear layers to a 58-unit hidden representation, increasing learned parameters from 216,250 to approximately 247,710 while retaining the proven architecture, paired-view objective, optimizer, and EMA.
mechanism: Wider flattened feature bottleneck
evidence_used: The best design uses only 216,250 of 250,000 permitted parameters, while deeper convolutional and attention alternatives regressed or timed out; widening the compact flattened head tests unused low-cost capacity without repeating those expensive architectural changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing label smoothing will exceed 9,286 correct predictions by accelerating class-boundary learning during the fixed two training exposures while EMA and dropout retain regularization.
change: Change the mirrored-ensemble loss from 2% label-smoothed cross-entropy to ordinary hard-label cross-entropy.
mechanism: Hard-target paired-view supervision
evidence_used: Probability-space view averaging lowered validation cross-entropy but reduced correct predictions from 9,286 to 9,280, showing that softer calibration does not improve the primary objective; hard targets directly test whether sharper discrimination raises accuracy.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 60.1488597500138, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.20767631034851075, "validation_score": 9265.414018223026}

RECENT RESULT
hypothesis: Expanding the classifier hidden representation from 48 to 54 units will exceed 9,286 correct predictions by preserving more spatial-feature information while remaining comfortably below the parameter and runtime costs of the timed-out 58-unit design.
change: Widen both classifier linear layers to 54 hidden units, increasing learned parameters from 216,250 to 235,132 while leaving the proven convolutional front end, paired-view objective, optimizer, schedule, and EMA unchanged.
mechanism: Moderate-width flattened feature bottleneck
evidence_used: The best design leaves 33,750 parameters unused, while deeper convolutional alternatives regressed and the 58-unit head timed out; 54 units uses only 18,882 additional low-cost classifier parameters and directly tests the capacity hypothesis with a smaller runtime increase.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 235132, "training_seconds": 70.63199012493715, "validation_accuracy": 0.9254, "validation_correct": 9254, "validation_cross_entropy": 0.22043434600830078, "validation_score": 9254.409690207127}

RECENT RESULT
hypothesis: Replacing the redundant pre-pair horizontal flip with cyclic spatial translations will exceed 9,286 correct predictions by adding genuine position variation without extra forward passes or runtime-heavy operations.
change: Cycle deterministically through all nine combinations of zero or one-pixel horizontal and vertical shifts using replicate padding; retain the proven mirrored-view objective, architecture, optimizer, schedule, and EMA.
mechanism: Cyclic one-pixel translation augmentation
evidence_used: Mirrored-pair training raised validation correct from 9,237 to 9,282, while auxiliary per-view supervision regressed. The current preparatory flip does not create a new transformation because the loss subsequently includes both horizontal orientations, so translation tests complementary augmentation while preserving the successful ensemble mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 216250, "training_seconds": 64.52828383306041, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.21675244331359864, "validation_score": 9244.410929932994}

RECENT RESULT
hypothesis: Lightweight channel attention will exceed 9,286 correct predictions by adaptively emphasizing useful final-stage features without the optimization cost of a wider classifier or global token attention.
change: Add a 1,096-parameter squeeze-and-excitation gate after the second pooling stage, centered around identity scaling, while preserving the proven training procedure.
mechanism: Identity-centered channel recalibration
evidence_used: Widening the classifier to 54 units reduced correct predictions to 9,254, and global-attention readout timed out; this tests targeted, computationally cheap feature adaptation instead of additional classifier capacity.
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
