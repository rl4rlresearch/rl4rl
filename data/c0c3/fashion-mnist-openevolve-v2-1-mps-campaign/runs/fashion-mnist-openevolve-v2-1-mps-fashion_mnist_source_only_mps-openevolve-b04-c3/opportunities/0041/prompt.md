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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.82347762514837, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2088966438293457, "validation_score": 9240.413600287959}
prior_hypothesis: A 12.5% terminal learning-rate floor will exceed 9,233 correct predictions by preserving the demonstrated benefit of late optimization while remaining well below the harmful 20% floor.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 50.31158974999562, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.20908836936950684, "validation_score": 9233.413534703224}
prior_hypothesis: A 7.5% terminal learning-rate floor will exceed 9,233 correct predictions by retaining useful late optimization while allowing more final convergence than the inferior 20% floor.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.61932904087007, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.210641943359375, "validation_score": 9227.413004028765}
prior_hypothesis: Restoring flip-only training and averaging flip predictions in probability space will exceed 9,229 correct predictions by preserving the qualified training procedure while producing a better-calibrated inference ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: A 13.125% terminal learning-rate floor will exceed 9,240 correct predictions by extending the improvement from 10% to 12.5% while remaining well below the harmful 20% floor.
change: Raise only the cosine schedule’s terminal floor from 12.5% to 13.125% of peak, preserving all other verified behavior.
mechanism: Fine-grained cosine-floor refinement
evidence_used: The 12.5% floor achieved the best verified result at 9,240 correct, improving over 9,233 at both 7.5% and 10%, while 20% fell to 9,226; the unverified 13.75% and 15% runs provide no accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Using four-channel grouped spatial filters with the verified 12.5% learning-rate floor will exceed 9,240 correct predictions by learning local cross-channel shape interactions that independent depthwise filters cannot represent.
change: Replace each depthwise residual convolution with a grouped 3×3 convolution whose filters jointly process four channels, increasing the model to 248,928 parameters, and use the strongest verified 12.5%-floor cosine schedule.
mechanism: Four-channel grouped spatial residual mixing
evidence_used: Schedule-only refinement plateaued at 9,240 correct with the 12.5% floor and degraded at 20%, motivating a representational change. All qualified architectures assume spatial filtering occurs independently per channel; the proposed grouped filters challenge that assumption with modest compute and remain below the parameter ceiling.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the exact strongest verified inference and schedule settings will increase correctness from 9,227 toward the demonstrated 9,240 predictions.
change: Replace probability-space flip fusion with arithmetic logit averaging and raise the cosine schedule’s terminal learning rate from zero to 12.5% of peak.
mechanism: Logit-space flip ensemble with 12.5%-floor cosine decay
evidence_used: Reference Design 2 achieved the best verified result—9,240 correct with 0.208897 cross-entropy—while the current probability-fusion, zero-floor design achieved 9,227 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Increasing the verified residual model’s classifier width from 38 to 39 will exceed 9,240 correct predictions by using the remaining parameter budget to preserve more spatial features without materially increasing convolutional runtime.
change: Restore the strongest verified 12.5%-floor residual depthwise design and widen its classifier bottleneck to 39 units, yielding approximately 249,757 learned parameters.
mechanism: Expanded low-cost classifier bottleneck
evidence_used: The 245,040-parameter reference achieved the best verified result at 9,240 correct, while added spatial mechanisms repeatedly timed out; widening the small dense head uses nearly all remaining capacity with negligible additional spatial computation.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249757, "training_seconds": 82.96516233379953, "validation_accuracy": 0.9223, "validation_correct": 9223, "validation_cross_entropy": 0.21159386291503907, "validation_score": 9223.412679541638}

RECENT RESULT
hypothesis: Adding a uniformly cycled ±2-pixel translation to flip augmentation will exceed 9,240 correct predictions by improving positional robustness without increasing parameters or model-forward cost.
change: Apply all 25 integer translations within a two-pixel radius evenly across training steps, using replicated padding while retaining per-image random horizontal flips.
mechanism: Balanced integer-translation augmentation
evidence_used: The 12.5%-floor model reached 9,240 correct, while further schedule refinement and classifier widening failed to improve it; translation augmentation is an orthogonal, computationally inexpensive change to the strongest verified design.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the terminal learning-rate floor from 7.5% to 12.5% will improve validation correctness from 9,233 toward the verified 9,240-result region.
change: Change only the cosine schedule’s terminal floor, preserving the architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.
mechanism: Verified 12.5%-floor cosine optimization
evidence_used: Reference Design 2 achieved the strongest verified result—9,240 correct with 0.208897 cross-entropy—using the identical implementation with a 12.5% floor, versus 9,233 correct for the current 7.5% floor.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact Reference Design 2 inference and schedule settings will improve validation correctness from 9,227 toward the demonstrated 9,240.
change: Replace probability-space flip fusion with arithmetic logit averaging and set the cosine schedule’s terminal learning rate to 12.5% of peak.
mechanism: Verified logit-space flip ensemble with nonzero-floor cosine decay
evidence_used: Reference Design 2 achieved the strongest verified result—9,240 correct with 0.208897 cross-entropy—while the current probability-fusion, zero-floor implementation achieved 9,227; unsuccessful replication attempts provide no contrary validation result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: The verified 245,040-parameter design will recover at least 9,240 correct predictions while reusing flipped stem features reduces validation computation enough to improve verification reliability.
change: Restore the strongest batch-64 residual depthwise model, flip augmentation, unsmoothed loss, and 12.5%-floor cosine schedule; compute the mathematically equivalent flip ensemble from one shared stem pass during evaluation.
mechanism: Flip-equivariant feature caching with residual depthwise classification
evidence_used: Reference Design 1 achieved the best result—9,240 correct and 0.208897 cross-entropy—but repeated full two-pass reproductions encountered verification failures or timeouts, motivating an inference-only efficiency improvement that preserves its learned architecture and ensemble.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Cycling through all nine ±1-pixel translations will exceed 9,240 correct predictions by improving positional robustness while avoiding the stronger perturbation and padding overhead of the timed-out ±2-pixel design.
change: Add a batch-shared, evenly cycled one-pixel translation via `torch.roll` before the existing random horizontal flip, preserving the strongest verified architecture and 12.5%-floor schedule.
mechanism: Low-overhead cyclic one-pixel translation augmentation
evidence_used: The current design achieved the best verified result at 9,240 correct, while the orthogonal ±2-pixel replicated-padding augmentation timed out; a smaller roll-based variant retains the promising invariance mechanism with less augmentation severity and overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the terminal learning-rate floor from 7.5% to 12.5% will improve correctness from 9,233 toward the verified 9,240-result region.
change: Change only the cosine schedule’s terminal floor, preserving the verified architecture, augmentation, optimizer, batch size, and logit-space flip ensemble.
mechanism: Verified nonzero-floor cosine optimization
evidence_used: Reference Design 2 differs from the current implementation only in this floor and achieved 9,240 correct with 0.208897 cross-entropy, versus 9,233 correct for the current 7.5% floor; failed replications produced no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Matching the strongest verified design will increase validation correctness from 9,227 toward the demonstrated 9,240.
change: Replace probability-space flip fusion with arithmetic logit averaging and raise the cosine schedule’s terminal learning-rate floor from 0% to 12.5%.
mechanism: Logit-space flip ensemble with 12.5%-floor cosine decay
evidence_used: Reference Design 2 achieved the best verified result—9,240 correct with 0.208897 cross-entropy—using these settings, while the current probability-fusion, zero-floor design achieved 9,227.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing the fixed-position 4,704-to-38 bottleneck with four learned spatial part summaries plus a global summary will exceed 9,240 correct predictions by preserving channel-rich local evidence while retaining the verified backbone’s runtime profile.
change: Restore the verified depthwise residual backbone, flip training and inference, and 12.5%-floor schedule, but compute predictions through initialized quadrant-attention pools and a 372-unit head; the model has 249,774 learned parameters.
mechanism: Content-adaptive multi-part attention pooling
evidence_used: The fixed-flatten design reached 9,240 correct, while widening its narrow classifier fell to 9,223 and added spatial convolutions timed out; reallocating the same head compute to content-dependent part aggregation cleanly challenges the shared fixed-location bottleneck without increasing convolutional work.
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
