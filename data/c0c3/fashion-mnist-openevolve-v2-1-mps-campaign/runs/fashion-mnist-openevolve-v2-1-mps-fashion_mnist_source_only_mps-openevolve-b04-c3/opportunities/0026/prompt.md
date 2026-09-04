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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 50.31158974999562, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.20908836936950684, "validation_score": 9233.413534703224}
prior_hypothesis: A 7.5% terminal learning-rate floor will exceed 9,233 correct predictions by retaining useful late optimization while allowing more final convergence than the inferior 20% floor.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.82347762514837, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2088966438293457, "validation_score": 9240.413600287959}
prior_hypothesis: A 12.5% terminal learning-rate floor will exceed 9,233 correct predictions by preserving the demonstrated benefit of late optimization while remaining well below the harmful 20% floor.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.61932904087007, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.210641943359375, "validation_score": 9227.413004028765}
prior_hypothesis: Restoring flip-only training and averaging flip predictions in probability space will exceed 9,229 correct predictions by preserving the qualified training procedure while producing a better-calibrated inference ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring flip-only training and averaging flip predictions in probability space will exceed 9,229 correct predictions by preserving the qualified training procedure while producing a better-calibrated inference ensemble.
change: Remove the harmful random translations and replace logit averaging at validation with an arithmetic mixture of class probabilities represented as log-probabilities.
mechanism: Probability-mixture flip ensemble
evidence_used: Flip-only batch-64 training achieved 9,229 correct, whereas adding translations reduced correctness to 9,058; probability-space fusion improves the existing flip ensemble without changing training cost or parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.61932904087007, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.210641943359375, "validation_score": 9227.413004028765}

RECENT RESULT
hypothesis: Reducing classifier dropout from 0.10 to 0.05 will exceed 9,229 correct predictions by retaining useful head regularization while reducing stochastic feature loss during the fixed two-pass training budget.
change: Change only the classifier dropout probability from 0.10 to 0.05, preserving the qualified architecture, batch size, augmentation, optimizer, schedule, and logit-space flip ensemble.
mechanism: Light bottleneck dropout
evidence_used: The qualified 0.10-dropout model achieved 9,229 correct, no dropout remained close at 9,225 with slightly better cross-entropy, and stronger 0.15 dropout fell to 9,190; this motivates testing lighter—not stronger—regularization.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring logit-space flip averaging and increasing the peak learning rate by 20% will exceed 9,229 correct predictions by translating the proven benefit of more optimizer updates into greater optimization distance without the timeout risk of smaller batches.
change: Restore the qualified logit-space flip ensemble and raise the AdamW cosine schedule’s peak learning rate from 2.5e-3 to 3.0e-3 while retaining batch size 64 and all other training behavior.
mechanism: Higher-step-equivalent cosine optimization
evidence_used: Batch size 64 improved correctness from 9,210 to 9,229 by doubling optimizer steps, but smaller batches timed out; probability-space fusion then reduced correctness to 9,227. This motivates returning to qualified inference and testing a compute-neutral increase in optimization distance.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining 10% of the peak learning rate at the end of the qualified batch-64 schedule will exceed 9,229 correct predictions by making the late optimizer steps useful without increasing runtime or peak-step instability.
change: Restore the qualified batch size of 64 and change cosine decay from zero-terminal to a 2.5e-4 learning-rate floor.
mechanism: Nonzero-floor cosine optimization
evidence_used: Batch size 64 achieved 9,229 correct versus 9,210 at batch size 128, showing that additional optimization updates help; a low cosine floor increases optimization distance without the added steps that caused smaller batches to time out.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.33236183295958, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.20886819381713867, "validation_score": 9233.413610021802}

RECENT RESULT
hypothesis: Raising the terminal learning-rate floor from 10% to 20% of peak will exceed 9,233 correct predictions by extracting more useful optimization from late steps without adding runtime or increasing the peak learning rate.
change: Increase the cosine schedule’s minimum learning rate from 2.5e-4 to 5.0e-4 while preserving the qualified model, batch size, augmentation, optimizer, and inference ensemble.
mechanism: Higher-floor cosine optimization
evidence_used: A 10% learning-rate floor improved the zero-terminal schedule from 9,229 to 9,233 correct predictions with lower cross-entropy, directly motivating a conservative test of whether additional late-stage optimization remains beneficial.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 76.95819154102355, "validation_accuracy": 0.9226, "validation_correct": 9226, "validation_cross_entropy": 0.2120160804748535, "validation_score": 9226.412535780717}

RECENT RESULT
hypothesis: A 7.5% terminal learning-rate floor will exceed 9,233 correct predictions by retaining useful late optimization while allowing more final convergence than the inferior 20% floor.
change: Change the zero-terminal cosine schedule to decay from 2.5e-3 to 1.875e-4, preserving all other qualified behavior.
mechanism: Fine-grained nonzero-floor cosine optimization
evidence_used: The 10% floor improved correctness from 9,229 to 9,233 and reduced cross-entropy, while increasing it to 20% fell to 9,226; testing 7.5% narrowly refines the supported region without changing runtime.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 50.31158974999562, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.20908836936950684, "validation_score": 9233.413534703224}

RECENT RESULT
hypothesis: Adding identity-initialized squeeze/excitation gates to the qualified 10%-floor, logit-ensemble model will exceed 9,233 correct predictions by letting global image context adaptively select residual channels without costly additional spatial processing.
change: Add global channel gates to both residual blocks, restore logit-space flip averaging, and use the best-performing 10% terminal learning-rate floor. The model has approximately 248,082 learned parameters.
mechanism: Globally conditioned residual channel gating
evidence_used: The 10%-floor logit-ensemble design achieved 9,233 correct, while added spatial computation repeatedly risked timeout and the alternative pooled head could not be verified. This challenges the shared assumption that locally computed channels should be applied identically to every image using only 3,042 additional parameters and negligible spatial work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 12.5% terminal learning-rate floor will exceed 9,233 correct predictions by preserving the demonstrated benefit of late optimization while remaining well below the harmful 20% floor.
change: Raise only the cosine schedule’s terminal learning-rate floor from 10% to 12.5% of peak, leaving architecture, augmentation, optimizer, batch size, and inference unchanged.
mechanism: Fine-grained upper-side cosine-floor refinement
evidence_used: The 10% floor achieved 9,233 correct with better cross-entropy than the tied 7.5% floor, while the 20% floor fell to 9,226; 12.5% is the closest untested refinement in the direction favored by that evidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 77.82347762514837, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2088966438293457, "validation_score": 9240.413600287959}

RECENT RESULT
hypothesis: A 15% terminal learning-rate floor will exceed 9,240 correct predictions by extending the improvement from 10% to 12.5% while remaining below the harmful 20% floor.
change: Raise only the cosine schedule’s terminal learning-rate floor from 7.5% to 15% of peak, preserving the qualified architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.
mechanism: Upper-bracket cosine-floor refinement
evidence_used: The 12.5% floor achieved the best result at 9,240 correct, improving over 9,233 at 10%, whereas 20% fell to 9,226; 15% is the most informative untested point between the current optimum and the demonstrated upper failure region.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified logit-space flip ensemble and 12.5% terminal learning-rate floor will reproduce the strongest available design, exceeding the current 9,227 correct predictions toward the demonstrated 9,240.
change: Replace probability-space flip fusion with arithmetic logit averaging and change zero-terminal cosine decay to a 12.5%-floor schedule.
mechanism: Logit-space flip ensemble with 12.5% cosine floor
evidence_used: Reference Design 2 achieved the best available result—9,240 correct with 0.208897 cross-entropy—using these settings; the current probability-mixture, zero-floor design achieved only 9,227 correct.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reproducing the strongest verified design will increase validation correctness from 8,928 toward the demonstrated 9,240 under the fixed exposure budget.
change: Replace the baseline CNN with the verified 245,040-parameter residual depthwise model, use batch size 64, random horizontal-flip training, logit-space flip ensembling, unsmoothed cross-entropy, and a 12.5%-floor cosine schedule.
mechanism: Residual depthwise CNN with flip invariance and nonzero-floor cosine optimization
evidence_used: Reference Design 1 achieved the best available result: 9,240 correct and 0.208897 cross-entropy; its 12.5% learning-rate floor also outperformed the verified 10%, 7.5%, and 20% floors.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 13.75% terminal learning-rate floor will exceed 9,240 correct predictions by extending the verified improvement from 10% to 12.5% while remaining closer to the current optimum than the unverified 15% setting and well below the harmful 20% floor.
change: Raise only the cosine schedule’s terminal learning-rate floor from 12.5% to 13.75% of peak, preserving all other training and inference behavior.
mechanism: Conservative upper-side cosine-floor refinement
evidence_used: The 12.5% floor achieved the best verified result at 9,240 correct, improving over 9,233 at 10%, while 20% fell to 9,226; the attempted 15% run timed out and supplied no validation evidence, making 13.75% the closest informative upper-side refinement.
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
