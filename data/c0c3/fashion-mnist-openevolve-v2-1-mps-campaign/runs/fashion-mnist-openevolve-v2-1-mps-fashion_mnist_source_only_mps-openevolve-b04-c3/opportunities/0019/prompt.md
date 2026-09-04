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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.16630454198457, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.21002658920288086, "validation_score": 9229.413214060303}
prior_hypothesis: Preserving the qualified 245,040-parameter residual model while reducing batch size from 128 to 64 will exceed 9,210 correct predictions because the fixed exposure budget will provide roughly 1,563 optimizer updates instead of 782.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 70.33236183295958, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.20886819381713867, "validation_score": 9233.413610021802}
prior_hypothesis: Retaining 10% of the peak learning rate at the end of the qualified batch-64 schedule will exceed 9,229 correct predictions by making the late optimizer steps useful without increasing runtime or peak-step instability.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.61932904087007, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.210641943359375, "validation_score": 9227.413004028765}
prior_hypothesis: Restoring flip-only training and averaging flip predictions in probability space will exceed 9,229 correct predictions by preserving the qualified training procedure while producing a better-calibrated inference ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding a near-capacity residual MLP to the qualified batch-64 model will exceed 9,229 correct predictions while completing within the time limit because it adds feature refinement with negligible spatial computation.
change: Use batch size 64 and spend 4,864 remaining parameters on a pre-normalized 38→63→38 residual MLP, bringing the model to 249,904 parameters.
mechanism: Compute-light residual classifier refinement
evidence_used: Reference Design 2 achieved 9,229 correct at batch size 64, while adding a residual block at 14×14 timed out; placing residual capacity in the compact classifier tests additional depth without that high-resolution runtime cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the qualified augmentation while reducing batch size from 64 to 56 will exceed 9,229 correct predictions by increasing optimizer steps from 1,564 to roughly 1,786 without the runtime risk observed at batch sizes 48 and 32.
change: Remove the harmful random-translation augmentation and use batch size 56 with the otherwise qualified residual design.
mechanism: Near-safe smaller-batch residual learning
evidence_used: Batch size 64 achieved 9,229 correct, improving on batch size 128’s 9,210; batch size 48 timed out, while translations at batch size 64 reduced correctness to 9,058. This motivates a conservative step toward more updates while restoring flip-only augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the final trajectory of the qualified batch-64 model will exceed 9,229 correct predictions by reducing small-batch parameter variance without increasing examples or spatial computation enough to risk timeout.
change: Maintain an exponential moving average of trainable parameters, update it every four optimizer steps, and install the averaged parameters after the final step; retain the qualified architecture, augmentation, loss, batch size, and learning-rate schedule.
mechanism: Low-overhead exponential weight averaging
evidence_used: Batch size 64 improved correctness from 9,210 to 9,229, while attempts to gain more updates at batch sizes 56, 48, and 32 timed out; parameter averaging targets the noisier small-batch trajectory without adding forward or backward passes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: With the qualified batch-64 encoder and training procedure, replacing the position-specific 7×7 flattening bottleneck with 2×2 average/max pooling and a 235-unit fusion layer will exceed 9,229 correct predictions by learning broader shape-and-presence interactions without materially increasing runtime.
change: Challenge the assumption that preserving every final spatial coordinate through a narrow 38-unit layer is the best parameter use; instead, aggregate each feature channel into coarse average and maximum maps, then classify their concatenation with a much wider head. The resulting model has approximately 249,329 learned parameters.
mechanism: Dual-statistic coarse spatial pooling
evidence_used: Reference Design 2 established 9,229 correct predictions for this encoder at batch size 64, while adding spatial residual computation and compute-light capacity on top both timed out. This replacement keeps encoder computation unchanged and substitutes—not adds—a similarly sized classifier with a different spatial representation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring flip-only augmentation and delaying cosine decay for the first 20% of training will exceed 9,229 correct predictions by increasing useful optimization distance without adding steps or meaningful runtime.
change: Remove the harmful random translations and retain the peak learning rate for 20% of training before annealing smoothly to zero.
mechanism: Delayed cosine annealing for stronger fixed-exposure optimization
evidence_used: The qualified batch-64 design reached 9,229 correct after increased optimizer steps, while translation augmentation fell to 9,058; this motivates preserving its data pipeline and computational cost while modestly increasing the learning-rate integral.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing head dropout will exceed 9,229 correct predictions by improving feature utilization during the fixed two-pass exposure budget without increasing runtime or parameters.
change: Replace the classifier’s 10% dropout with an identity operation while retaining the qualified architecture, batch size, augmentation, optimizer, and schedule.
mechanism: Deterministic low-exposure classifier head
evidence_used: The qualified batch-64 model improved from 9,210 to 9,229 correct through additional optimization updates, while added augmentation reduced accuracy and added computation repeatedly timed out; removing stochastic head regularization directly targets limited-exposure fitting at lower computational cost.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 71.73461883398704, "validation_accuracy": 0.9225, "validation_correct": 9225, "validation_cross_entropy": 0.2098698944091797, "validation_score": 9225.413267577209}

RECENT RESULT
hypothesis: Increasing classifier dropout from 0.10 to 0.15 in the qualified batch-64 model will exceed 9,229 correct predictions by modestly strengthening the head regularization that already outperformed no dropout.
change: Restore the qualified batch size of 64 and increase the existing classifier dropout probability to 0.15 without changing parameters or computational structure.
mechanism: Stronger bottleneck dropout
evidence_used: Reference Design 2 achieved 9,229 correct with 10% dropout, while removing dropout reduced correctness to 9,225 despite slightly lower cross-entropy; this directly motivates testing a small increase in dropout to prioritize the accuracy-ranked objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 80.22242033388466, "validation_accuracy": 0.919, "validation_correct": 9190, "validation_cross_entropy": 0.21593998069763184, "validation_score": 9190.41120450675}

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
