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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 53.25754679203965, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.22417662353515624, "validation_score": 9290.40843779434}
prior_hypothesis: Expanding the dense bottleneck from 48 to 58 units will exceed 9,265 correct predictions by using the remaining parameter budget to improve class separation without altering the successful augmentation and optimization procedure.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Expanding the dense bottleneck from 48 to 58 units will exceed 9,265 correct predictions by using the remaining parameter budget to improve class separation without altering the successful augmentation and optimization procedure.
change: Increase the classifier bottleneck width to 58 units, raising learned parameters from 216,346 to 247,836 while retaining all other settings.
mechanism: Near-ceiling dense bottleneck expansion
evidence_used: Reallocating dense-head capacity into a fifth convolution reduced validation correct from 9,177 to 9,157; preserving the successful convolutional extractor and adding capacity to the dense head is therefore the most direct untested use of the 250,000-parameter ceiling.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 53.25754679203965, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.22417662353515624, "validation_score": 9290.40843779434}

RECENT RESULT
hypothesis: Adding identity skips around the second convolution of each stage will exceed 9,290 correct predictions by improving feature preservation and gradient flow while retaining the verified 247,836-parameter head, augmentation, optimizer, and ensemble.
change: Replace the sequential convolutional extractor with equivalent explicitly named layers and add residual additions within both constant-width convolutional stages.
mechanism: Parameter-free intra-stage residual connections
evidence_used: Expanding the existing dense bottleneck improved validation correct from 9,265 to 9,290, whereas replacing head capacity with a fifth convolution regressed; this motivates preserving all successful capacity while testing parameter-free connectivity in the existing extractor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 54.66474029212259, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.22120282135009767, "validation_score": 9273.409432398335}

RECENT RESULT
hypothesis: Adding a 2,128-parameter channel gate will exceed 9,290 correct predictions by adaptively recalibrating the final convolutional features while preserving the verified architecture’s initial behavior and staying below the parameter ceiling.
change: Insert an identity-initialized 64→16→64 channel gate before the final pooling layer, increasing learned parameters from 247,836 to 249,964.
mechanism: Identity-initialized squeeze-and-excitation channel gating
evidence_used: Dense-head expansion improved validation correct from 9,265 to 9,290, while reallocating capacity to a fifth convolution regressed; this motivates using the remaining parameter budget for feature refinement without removing successful capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249964, "training_seconds": 70.95072883390822, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.2237412094116211, "validation_score": 9268.408583118844}

RECENT RESULT
hypothesis: Sampling the centered view 50% of the time and each cardinal translation 12.5% will exceed 9,290 correct predictions by concentrating limited training exposure on the validation distribution while retaining useful one-pixel invariance.
change: Reweight training augmentation from uniform sampling across five views to a 4:1:1:1:1 center-to-cardinal distribution; evaluation and all other settings remain unchanged.
mechanism: Center-biased cardinal-translation augmentation
evidence_used: Restricting augmentation from 25 translations to the centered and four cardinal views increased validation correct from 9,188 to 9,265, the largest recent gain; further concentrating exposure on the unshifted validation geometry directly extends that successful reduction in augmentation strength.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 75.04214420798235, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.22305315475463866, "validation_score": 9278.408812975998}

RECENT RESULT
hypothesis: A learned 64→48 pointwise projection followed by a 76-unit dense bottleneck will exceed 9,290 correct predictions by providing greater nonlinear head width within the parameter ceiling while retaining the successful convolutional extractor.
change: Insert a normalized pointwise channel projection before flattening and widen the classifier bottleneck from 58 to 76 units, increasing the model to 248,102 learned parameters while leaving training, augmentation, optimization, and ensembling unchanged.
mechanism: Channel-projected wider dense bottleneck
evidence_used: Expanding the dense bottleneck from 48 to 58 improved validation correct from 9,265 to 9,290, whereas adding a fifth convolution while narrowing the bottleneck to 36 regressed to 9,157; this motivates a parameter-efficient way to extend dense-head width without replacing the verified extractor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248102, "training_seconds": 66.64561574999243, "validation_accuracy": 0.9272, "validation_correct": 9272, "validation_cross_entropy": 0.2224986503601074, "validation_score": 9272.408998406545}

RECENT RESULT
hypothesis: Removing dense-head dropout will exceed 9,290 correct predictions by allowing the near-ceiling 58-unit bottleneck to use its full capacity during the limited two-pass training exposure.
change: Replace the classifier’s 10% dropout with an identity operation while preserving its architecture, parameter count, augmentation, optimizer, schedule, and EMA ensemble.
mechanism: Dense-head dropout ablation
evidence_used: Expanding the dense bottleneck from 48 to 58 units improved validation correct from 9,265 to 9,290, while convolutional reallocation, channel gating, and projected widening regressed; this indicates that preserving and more fully optimizing the successful dense representation is the most relevant remaining lever.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 65.86824408289976, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.2234785961151123, "validation_score": 9270.408670819079}

RECENT RESULT
hypothesis: Ramping dense-head dropout from 5% to 15% while preserving its 10% average will exceed 9,290 correct predictions by easing early optimization without losing the regularization whose removal reduced accuracy to 9,270.
change: Replace fixed 10% training dropout with a linear 5%→15% schedule over the fixed exposure.
mechanism: Late-strengthened dropout curriculum
evidence_used: Removing 10% dropout reduced validation correct from 9,290 to 9,270, showing that dropout is useful; scheduling the same average strength tests whether its early optimization cost can be reduced without sacrificing late regularization.
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
