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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 54.925307208905, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669889450073242, "validation_score": 9289.414353574266}
prior_hypothesis: Scaling the verified 10% pair-wide penalty up toward 25% only when flip predictions have large probability-space disagreement will exceed 9,289 correct predictions while avoiding the calibration cost of uniformly applying the stronger penalty.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.03452316601761, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.2067018039703369, "validation_score": 9289.414352575222}
prior_hypothesis: A 15% penalty on flip-inconsistent offset pairs will exceed 9,287 correct predictions by suppressing locally unreliable transformed evidence while leaving flip-stable pooling unchanged.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.99014087487012, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669277229309083, "validation_score": 9289.414355676507}
prior_hypothesis: Restoring the verified 10% flip-inconsistency penalty and mildly reducing confidence for intermediate-agreement ensembles will retain 9,289 correct predictions while lowering validation cross-entropy below 0.20669593772888184.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.64032600005157, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669593772888184, "validation_score": 9289.414354589559}
prior_hypothesis: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.

## Recent verification evidence

RECENT RESULT
hypothesis: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.
change: Replace fixed offset pooling with per-image flip-consistency weights of 0.90 for inconsistent pairs and 1.0 for consistent pairs, retaining the verified center preference and agreement calibration.
mechanism: Lower-dose flip-consistency-weighted logit ensembling
evidence_used: Reducing the penalty from 15% to 12.5% retained 9,289 correct and improved cross-entropy from 0.2067018039703369 to 0.2066986053466797, while increasing it to 25% worsened cross-entropy; testing 10% probes the minimum effective penalty.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.64032600005157, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669593772888184, "validation_score": 9289.414354589559}

RECENT RESULT
hypothesis: A 7.5% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.20669593772888184.
change: Replace fixed offset pooling with per-image weights of 0.925 for flip-inconsistent pairs and 1.0 for consistent pairs, preserving center preference and agreement calibration.
mechanism: Lower-dose flip-consistency-weighted logit ensembling
evidence_used: Reducing the penalty from 15% to 12.5% to 10% retained 9,289 correct while successively lowering cross-entropy; 7.5% is the next dose toward the minimum effective penalty.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 51.79009729181416, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20669369468688964, "validation_score": 9288.414355359775}

RECENT RESULT
hypothesis: An 8.75% flip-inconsistency penalty will retain 9,289 correct predictions while lowering validation cross-entropy below 0.20669593772888184.
change: Reduce the inconsistent offset-pair penalty from 10% to 8.75%, leaving training, center preference, and agreement calibration unchanged.
mechanism: Boundary-seeking flip-consistency weighting
evidence_used: Penalties of 15%, 12.5%, and 10% all retained 9,289 correct while lower penalties progressively improved cross-entropy; 7.5% lost one correct prediction, placing the useful decision boundary between 7.5% and 10%.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 74.78914975002408, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20669475784301758, "validation_score": 9288.414354994708}

RECENT RESULT
hypothesis: Replacing both hard max-pooling stages with learned per-channel mixtures of average and maximum pooling will exceed 9,289 correct predictions by reducing one-pixel phase sensitivity while retaining strong localized activations.
change: Add channel-adaptive mixed pooling at both resolution reductions; remove the redundant affine transform from the classifier’s LayerNorm to remain below the parameter ceiling.
mechanism: Channel-adaptive mixed downsampling
evidence_used: Flip-consistency weighting improved correctness from 9,287 to 9,289, indicating sensitivity to transformed evidence. Unlike the failed spatial attention and static spatial pooling designs, this mechanism neither selects absolute locations nor discards channels; it challenges the load-bearing assumption that every feature channel benefits from the same hard maximum downsampling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249673, "training_seconds": 72.5778272501193, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20190465087890624, "validation_score": 9280.416006377573}

RECENT RESULT
hypothesis: A 9.375% flip-inconsistency penalty will retain 9,289 correct predictions while reducing validation cross-entropy below the 10% penalty’s 0.20669593772888184.
change: Reduce the inconsistent offset-pair penalty from 15% to 9.375%, leaving training, center preference, and agreement calibration unchanged.
mechanism: Boundary-bisecting flip-consistency weighting
evidence_used: Penalties of 10%–15% retained 9,289 correct, while 8.75% lost one; 9.375% bisects the established decision boundary and tests whether the lower-cross-entropy dose can preserve the accuracy gain.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.62712899991311, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.2066953327178955, "validation_score": 9288.414354797307}

RECENT RESULT
hypothesis: A 9.6875% flip-inconsistency penalty will retain 9,289 correct predictions while lowering validation cross-entropy below the 10% penalty’s 0.20669593772888184.
change: Increase the current penalty from 7.5% to 9.6875%, bisecting the established boundary between the unsuccessful 9.375% dose and successful 10% dose.
mechanism: Boundary-bisecting flip-consistency weighting
evidence_used: The 10% penalty achieved 9,289 correct, while 9.375% achieved 9,288; 9.6875% is the most informative untested point between them.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.14332745899446, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20669563064575194, "validation_score": 9288.414354695005}

RECENT RESULT
hypothesis: A 9.84375% flip-inconsistency penalty will retain 9,289 correct predictions while lowering validation cross-entropy below 0.20669593772888184.
change: Reduce the inconsistent offset-pair penalty from 10% to 9.84375%, leaving training, center preference, and agreement calibration unchanged.
mechanism: Boundary-bisecting flip-consistency weighting
evidence_used: A 10% penalty retained 9,289 correct, whereas 9.6875% produced 9,288; 9.84375% bisects this established decision boundary and is the closest untested lower penalty likely to preserve correctness while improving calibration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.53330587502569, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20669578361511232, "validation_score": 9288.414354642478}

RECENT RESULT
hypothesis: A 9.921875% flip-inconsistency penalty will retain 9,289 correct predictions while lowering cross-entropy below the 10% penalty’s 0.20669593772888184.
change: Reduce the current 12.5% penalty to 9.921875%, leaving training, center preference, and agreement calibration unchanged.
mechanism: Boundary-bisecting flip-consistency weighting
evidence_used: A 10% penalty achieved 9,289 correct, while 9.84375% achieved 9,288; 9.921875% bisects the established correctness boundary and is the most informative lower penalty likely to retain the extra correct prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 50.74439470889047, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20669585838317872, "validation_score": 9288.414354616805}

RECENT RESULT
hypothesis: Penalizing only the consensus-dissenting member of each flip-inconsistent pair will exceed 9,289 correct predictions by preserving reliable transformed evidence while suppressing the specific outlier view.
change: Replace pair-wide penalties with a 20% penalty on dissenting individual views, preserving the successful 10% total pair-weight reduction when one member agrees with the initial ensemble.
mechanism: Consensus-directed flip-pair reweighting
evidence_used: A 10% pair-wide penalty achieved 9,289 correct, while agreement-conditioned calibration showed that ensemble consensus carries reliability information; this combines both signals without probability-space pooling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 60.72388787497766, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.20669459419250488, "validation_score": 9287.414355050902}

RECENT RESULT
hypothesis: Scaling the verified 10% pair-wide penalty up toward 25% only when flip predictions have large probability-space disagreement will exceed 9,289 correct predictions while avoiding the calibration cost of uniformly applying the stronger penalty.
change: Restore the successful 10% minimum penalty and continuously increase it to at most 25% using total-variation disagreement between each offset’s original and flipped predictions.
mechanism: Disagreement-adaptive flip-pair weighting
evidence_used: Uniform 10% and 25% penalties both achieved 9,289 correct, but 25% worsened cross-entropy; meanwhile, penalizing individual dissenting views fell to 9,287. This motivates retaining proven pair-wide suppression while reserving stronger weighting for the most unreliable pairs.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 54.925307208905, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669889450073242, "validation_score": 9289.414353574266}

RECENT RESULT
hypothesis: Retaining the successful 10% penalty but increasing it to 25% only for flip-inconsistent offset pairs whose averaged prediction opposes the provisional ensemble will exceed 9,289 correct predictions.
change: Use center-weighted provisional consensus to distinguish supportive from dissenting flip-inconsistent pairs, applying a 10% penalty to the former and 25% to the latter.
mechanism: Consensus-gated flip-pair suppression
evidence_used: A uniform 10% pair-wide penalty achieved 9,289 correct, while uniform 25% retained that count but worsened cross-entropy and penalizing individual dissenting views fell to 9,287; pair-level semantic gating tests stronger suppression only where both inconsistency and ensemble opposition indicate unreliable evidence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 55.67252695793286, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20667744255065917, "validation_score": 9288.41436094052}

RECENT RESULT
hypothesis: Restoring the verified 10% flip-inconsistency penalty and mildly reducing confidence for intermediate-agreement ensembles will retain 9,289 correct predictions while lowering validation cross-entropy below 0.20669593772888184.
change: Use the accuracy-preserving 10% pair penalty and add a small quadratic uncertainty correction to validation-logit calibration without changing predicted classes.
mechanism: Agreement-curved confidence calibration
evidence_used: The 10% penalty achieved 9,289 correct with the best reported cross-entropy, while every tested lower penalty lost one prediction and stronger/adaptive suppression did not improve the objective; this motivates preserving its decisions and refining only confidence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.99014087487012, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669277229309083, "validation_score": 9289.414355676507}



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
