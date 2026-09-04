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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 55.24326145788655, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.20867646484375, "validation_score": 9277.413675631604}
prior_hypothesis: Restoring AdamW beta2=0.96 and increasing the cosine schedule’s peak learning rate by 5% will exceed 9,283 correct predictions by accelerating convergence during the fixed 522-step run without materially destabilizing the proven model.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 34.46409162483178, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20936809196472167, "validation_score": 9283.4134390541}
prior_hypothesis: AdamW beta2=0.96 will exceed 9,278 correct predictions by continuing the verified improvement from beta2=0.99 through 0.98 to 0.97 while remaining stable over 522 optimizer steps.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.22171137481928, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20909543914794923, "validation_score": 9280.41353228522}
prior_hypothesis: AdamW beta2=0.95 will exceed 9,283 correct predictions by continuing the verified monotonic improvement from beta2=0.99 through 0.96 while remaining stable over 522 optimizer steps.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 42.50887762499042, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20954851112365722, "validation_score": 9278.413377384539}
prior_hypothesis: AdamW beta2=0.97 will exceed 9,277 correct predictions by continuing the improvement observed from beta2=0.99 to 0.98 while remaining stable over the fixed 522-step run.

## Recent verification evidence

RECENT RESULT
hypothesis: AdamW beta2=0.99 will exceed 9,268 correct predictions by adapting gradient-variance estimates faster during the fixed 522-step training run while retaining the best verified architecture and TTA.
change: Restore Reference Design 3’s vertical-biased TTA and use a faster-decaying AdamW second-moment estimate.
mechanism: Short-horizon second-moment adaptation
evidence_used: Reference Design 3 achieved the best tied score with 9,268 correct; architecture, augmentation, and late-trajectory averaging changes regressed, motivating a clean optimization-dynamics test on the strongest verified representation.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 41.99500587489456, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21068120079040528, "validation_score": 9268.412990636736}

RECENT RESULT
hypothesis: AdamW beta2=0.98 will exceed 9,268 correct predictions by extending the improvement from beta2=0.99 while remaining stable over the fixed 522-step run.
change: Reduce AdamW beta2 from 0.99 to 0.98 while preserving the best architecture, schedule, augmentation, and vertical-biased TTA.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Changing beta2 from AdamW’s default 0.999 to 0.99 retained 9,268 correct and reduced validation cross-entropy from 0.2124414 to 0.2106812, the strongest verified optimizer improvement.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 30.580157082993537, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.2100638172149658, "validation_score": 9277.413201347637}

RECENT RESULT
hypothesis: AdamW beta2=0.97 will exceed 9,277 correct predictions by continuing the improvement observed from beta2=0.99 to 0.98 while remaining stable over the fixed 522-step run.
change: Reduce AdamW beta2 from its current default of 0.999 to 0.97 while preserving the strongest architecture, augmentation, schedule, and vertical-biased TTA.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Beta2=0.99 reduced cross-entropy to 0.2106812 without losing accuracy, and beta2=0.98 further improved to 9,277 correct with 0.2100638 cross-entropy, supporting a conservative continuation of the same optimizer trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 42.50887762499042, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20954851112365722, "validation_score": 9278.413377384539}

RECENT RESULT
hypothesis: AdamW beta2=0.96 will exceed 9,278 correct predictions by continuing the verified improvement from beta2=0.99 through 0.98 to 0.97 while remaining stable over 522 optimizer steps.
change: Restore the strongest vertical-biased TTA configuration and reduce AdamW beta2 to 0.96, preserving the architecture, augmentation, learning-rate schedule, and parameter count.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Beta2 reductions produced monotonic gains: 0.99 achieved 9,268 correct, 0.98 achieved 9,277, and 0.97 achieved 9,278 with the lowest cross-entropy of 0.2095485.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 34.46409162483178, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20936809196472167, "validation_score": 9283.4134390541}

RECENT RESULT
hypothesis: AdamW beta2=0.95 will exceed 9,283 correct predictions by continuing the verified monotonic improvement from beta2=0.99 through 0.96 while remaining stable over 522 optimizer steps.
change: Restore the strongest vertical-biased, distance-stratified TTA and reduce AdamW beta2 to 0.95 while preserving all other training and model settings.
mechanism: Faster short-horizon second-moment adaptation
evidence_used: Beta2 reductions improved validation from 9,268 correct at 0.99 to 9,277 at 0.98, 9,278 at 0.97, and 9,283 at 0.96, with cross-entropy decreasing monotonically to 0.2093681.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.22171137481928, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20909543914794923, "validation_score": 9280.41353228522}

RECENT RESULT
hypothesis: AdamW beta2=0.9575 will retain or exceed 9,283 correct predictions while lowering cross-entropy below 0.2093681.
change: Reduce AdamW beta2 from 0.96 to 0.9575 while preserving every other model, training, and evaluation setting.
mechanism: Bracketed second-moment decay interpolation
evidence_used: Beta2=0.96 achieved the best accuracy at 9,283 correct, while 0.95 lowered cross-entropy to 0.2090954 but lost three predictions; an interior value tests whether the cross-entropy gain can be captured without crossing the accuracy optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 46.96981837507337, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20925290603637695, "validation_score": 9280.413478435737}

RECENT RESULT
hypothesis: Adding a low-rank two-head spatial-attention block will exceed 9,283 correct predictions by learning content-dependent relationships among distant garment parts before spatial aggregation, while the residual scale preserves the strong convolutional baseline during short training.
change: Preserve the max-pooled convolutional representation, spatial classifier, augmentation, and TTA; insert compact relative-position self-attention at the 7×7 feature stage and restore the best verified AdamW beta2 of 0.96.
mechanism: Relative-position non-local part attention
evidence_used: Beta2=0.96 reached 9,283 correct, while further optimizer interpolation, pooling replacement, global channel gating, and head changes failed to improve it. This challenges the shared assumption that fixed local/dilated convolution alone adequately models long-range part relationships without discarding the verified backbone or classifier.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249489, "training_seconds": 40.61832020804286, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.20968514709472658, "validation_score": 9265.413330692867}

RECENT RESULT
hypothesis: AdamW beta2=0.9625 will exceed 9,283 correct predictions by refining the unexplored upper side of the narrow optimum around beta2=0.96.
change: Increase AdamW beta2 from 0.97 to 0.9625 while preserving all other model, training, augmentation, schedule, and evaluation settings.
mechanism: Symmetric upper-side second-moment interpolation
evidence_used: Beta2=0.96 achieved the best result at 9,283 correct, while the equally spaced lower-side value 0.9575 achieved 9,280 and beta2=0.97 achieved 9,278; testing 0.9625 completes the closest symmetric bracket around the observed optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 37.78091958304867, "validation_accuracy": 0.9272, "validation_correct": 9272, "validation_cross_entropy": 0.20921563911437988, "validation_score": 9272.413491178766}

RECENT RESULT
hypothesis: Restoring AdamW beta2=0.96 and increasing the cosine schedule’s peak learning rate by 5% will exceed 9,283 correct predictions by accelerating convergence during the fixed 522-step run without materially destabilizing the proven model.
change: Set AdamW beta2 to the best verified value of 0.96 and consistently raise the optimizer and scheduled base learning rate from 2.0e-3 to 2.1e-3.
mechanism: Peak learning-rate refinement at the verified beta2 optimum
evidence_used: Beta2=0.96 achieved the best result at 9,283 correct, while nearby beta2 values on both sides regressed; this fixes beta2 at its observed optimum and tests the next central short-horizon optimization variable with a conservative change.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 55.24326145788655, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.20867646484375, "validation_score": 9277.413675631604}

RECENT RESULT
hypothesis: AdamW beta2=0.96 with a 2.5% lower peak learning rate will exceed 9,283 correct predictions by reducing boundary-crossing update noise while retaining sufficient convergence over 522 steps.
change: Preserve the best verified architecture, augmentation, TTA, beta2, and cosine shape while lowering the optimizer and scheduled base learning rate from 2.0e-3 to 1.95e-3.
mechanism: Conservative peak learning-rate refinement
evidence_used: Beta2=0.96 at 2.0e-3 achieved the best 9,283 correct, whereas raising the learning rate to 2.1e-3 reduced cross-entropy but lost six correct predictions; a conservative lower-side test directly brackets the accuracy-optimal learning rate.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 59.50231920811348, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20959160385131836, "validation_score": 9279.413362657619}

RECENT RESULT
hypothesis: AdamW beta2=0.96 with a 2.025e-3 peak learning rate will retain or exceed 9,283 correct predictions while lowering cross-entropy below 0.2093681.
change: Restore the best verified beta2 and raise both the optimizer and cosine-schedule base learning rate by 1.25%.
mechanism: Upper-side peak-learning-rate interpolation
evidence_used: Beta2=0.96 at 2.0e-3 achieved the best 9,283 correct; 2.1e-3 lowered cross-entropy to 0.2086765 but lost six predictions, while 1.95e-3 worsened both accuracy and cross-entropy, motivating a narrow upper-side interpolation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 46.01391216600314, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.2093422607421875, "validation_score": 9275.413447885045}

RECENT RESULT
hypothesis: Restoring beta2=0.96 while annealing the learning rate fully to zero will exceed 9,283 correct predictions by preserving the accuracy-optimal peak rate and reducing boundary-crossing updates late in the fixed 522-step run.
change: Restore the best verified AdamW beta2 and replace the 5% cosine learning-rate floor with a zero floor.
mechanism: Zero-floor terminal cosine annealing
evidence_used: Beta2=0.96 at a 2.0e-3 peak achieved the best 9,283 correct; changing the peak in either direction regressed, so isolating the previously untested terminal learning rate is the most direct schedule refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 47.407488375203684, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.21438936767578126, "validation_score": 9252.411729559983}



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
