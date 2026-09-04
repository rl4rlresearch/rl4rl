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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 78.33630070788786, "validation_accuracy": 0.9058, "validation_correct": 9058, "validation_cross_entropy": 0.2554629264831543, "validation_score": 9058.398259470234}
prior_hypothesis: Adding inexpensive two-pixel random translations to the qualified batch-64 residual design will exceed 9,229 correct predictions by improving spatial invariance without the runtime increase that prevented smaller batches from finishing.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245040, "training_seconds": 41.259548500180244, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.21916018409729005, "validation_score": 9210.410118380278}
prior_hypothesis: Using the available parameter headroom for batch-normalized residual features, while doubling optimizer updates and removing target smoothing, will exceed the baseline’s 8,928 correct predictions after the same 100,000 examples.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.16630454198457, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.21002658920288086, "validation_score": 9229.413214060303}
prior_hypothesis: Preserving the qualified 245,040-parameter residual model while reducing batch size from 128 to 64 will exceed 9,210 correct predictions because the fixed exposure budget will provide roughly 1,563 optimizer updates instead of 782.

## Recent verification evidence

RECENT RESULT
hypothesis: Preserving the qualified 245,040-parameter residual model while reducing batch size from 128 to 64 will exceed 9,210 correct predictions because the fixed exposure budget will provide roughly 1,563 optimizer updates instead of 782.
change: Adopt the qualified residual depthwise architecture, flip augmentation, flip-averaged inference, and unsmoothed loss, while halving its batch size to 64.
mechanism: Smaller-batch residual depthwise learning
evidence_used: Reference Design 1 improved validation correctness from 8,928 to 9,210 while increasing optimizer steps from 392 to 782; this motivates testing whether another update-count increase improves the same established model.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 63.16630454198457, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.21002658920288086, "validation_score": 9229.413214060303}

RECENT RESULT
hypothesis: Halving the qualified residual model’s batch size from 64 to 32 will exceed 9,229 correct predictions because the prior 128-to-64 reduction increased correctness while holding architecture and exposure fixed.
change: Adopt the 245,040-parameter residual depthwise design with flip augmentation and flip-averaged inference, using batch size 32 for roughly twice as many optimizer updates as Reference Design 2.
mechanism: Higher-frequency smaller-batch residual learning
evidence_used: Reference Design 2 improved from 9,210 to 9,229 correct solely by reducing batch size from 128 to 64, motivating a controlled continuation of the observed update-count trend.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The qualified residual model with batch size 48 will complete within the verification limit and exceed 9,229 correct predictions by increasing optimizer steps from 1,564 to roughly 2,084 without the runtime cost that prevented batch size 32 from finishing.
change: Adopt the 245,040-parameter residual depthwise architecture, flip augmentation, flip-averaged inference, and unsmoothed cross-entropy from Reference Design 2, while using batch size 48.
mechanism: Intermediate-batch residual depthwise learning
evidence_used: Reducing batch size from 128 to 64 improved correctness from 9,210 to 9,229, while batch size 32 failed to finish; batch size 48 tests the useful middle ground between additional updates and verification runtime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding inexpensive two-pixel random translations to the qualified batch-64 residual design will exceed 9,229 correct predictions by improving spatial invariance without the runtime increase that prevented smaller batches from finishing.
change: Adopt Reference Design 2’s architecture, batch size, flip augmentation, flip-averaged inference, and unsmoothed loss, then add random padded crops during training.
mechanism: Random-translation residual learning
evidence_used: Reference Design 2 achieved 9,229 correct at batch size 64, while batch sizes 48 and 32 timed out; this motivates retaining the fastest qualified configuration and testing a low-overhead augmentation improvement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245040, "training_seconds": 78.33630070788786, "validation_accuracy": 0.9058, "validation_correct": 9058, "validation_cross_entropy": 0.2554629264831543, "validation_score": 9058.398259470234}

RECENT RESULT
hypothesis: Adding a second 48-channel residual depthwise block while retaining batch size 64 will exceed 9,229 correct predictions by using the remaining parameter headroom for deeper feature refinement.
change: Add one residual depthwise block at the 14×14 feature stage, increasing parameters from 245,040 to approximately 247,968 without changing the qualified optimizer, augmentation, or batch size.
mechanism: Additional mid-resolution residual refinement
evidence_used: The current batch-64 residual design achieved 9,229 correct with 245,040 parameters, while translation augmentation reduced correctness to 9,058; this motivates preserving its established training procedure and testing a focused capacity increase below the 250,000-parameter ceiling.
result: training did not finish within the verification time limit

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
