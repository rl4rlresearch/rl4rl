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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 78.51267020893283, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.1962520164489746, "validation_score": 9312.417972127214}
prior_hypothesis: Training every presented image alongside its horizontal mirror with a small symmetric consistency loss will reduce the flip instability that inference-time penalties identified, exceeding 9,289 correct predictions while retaining the best validated inference pooling and calibration.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.03452316601761, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.2067018039703369, "validation_score": 9289.414352575222}
prior_hypothesis: A 15% penalty on flip-inconsistent offset pairs will exceed 9,287 correct predictions by suppressing locally unreliable transformed evidence while leaving flip-stable pooling unchanged.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.75497637502849, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19384812965393067, "validation_score": 9312.418813739856}
prior_hypothesis: Training 75% of pairs as exact reflections and 25% as reflected pairs from different validation offsets will exceed 9,312 correct predictions without the four-view design’s timeout.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.64032600005157, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669593772888184, "validation_score": 9289.414354589559}
prior_hypothesis: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Training every presented image alongside its horizontal mirror with a small symmetric consistency loss will reduce the flip instability that inference-time penalties identified, exceeding 9,289 correct predictions while retaining the best validated inference pooling and calibration.
change: Restore the verified uniform 10% inconsistency penalty and agreement-curved calibration, then jointly train original/flipped pairs using smoothed classification loss plus symmetric KL consistency.
mechanism: Flip-paired consistency training
evidence_used: Uniform 10% pair suppression reached 9,289 correct, whereas adaptive or member-specific suppression did not improve correctness; this suggests reducing flip disagreement during learning is more promising than further inference-time reweighting.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 78.51267020893283, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.1962520164489746, "validation_score": 9312.417972127214}

RECENT RESULT
hypothesis: Exact reflection-invariant fusion of aligned late feature maps, while retaining absolute orientation-sensitive discrepancies, will exceed the 9,312 correct predictions of soft flip-consistency training.
change: Replace independent flip-logit averaging with a learned classifier over reflection-stable and reflection-sensitive feature statistics; evaluate only the five distinct offset orbits and retain the best validated agreement calibration.
mechanism: Reflection-orbit stable/sensitive feature fusion
evidence_used: Paired flip-consistency training improved correctness from 9,289 to 9,312, showing that reflection instability is load-bearing. The old designs assume augmentation, regularization, and output averaging are sufficient; this patch instead makes reflection-orbit structure part of the image representation without discarding informative asymmetric filter responses.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249812, "training_seconds": 78.594971582992, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20234075317382813, "validation_score": 9280.415855487456}

RECENT RESULT
hypothesis: Training each example with its reflected pair at two neighboring offsets, plus consistency between the offset ensembles, will exceed 9,312 correct predictions by extending the successful flip-paired regularization to the remaining transformations used at validation.
change: Add a cyclic one-pixel translated pair to every training batch, supervise all four views, retain the 5% within-offset flip consistency loss, and add a smaller 2.5% consistency loss between the two flip-averaged offsets.
mechanism: Reflection-and-translation orbit consistency training
evidence_used: Flip-paired consistency training improved correctness from 9,289 to 9,312, while learned reflection-orbit feature fusion fell to 9,280; this favors preserving the successful architecture and extending consistency training to validation-time offset variation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training 75% of pairs as exact reflections and 25% as reflected pairs from different validation offsets will exceed 9,312 correct predictions without the four-view design’s timeout.
change: Move offset augmentation into the loss, supervise two views per example, and apply 5% consistency to exact flip pairs versus 2.5% consistency to cross-offset pairs.
mechanism: Compute-neutral stochastic flip-offset orbit pairing
evidence_used: Exact flip-paired training improved validation correctness from 9,289 to 9,312, while extending every example to four reflection-and-translation views exceeded the time limit; stochastic two-view pairing tests the same invariance within the successful compute budget.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.75497637502849, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19384812965393067, "validation_score": 9312.418813739856}

RECENT RESULT
hypothesis: Making every cross-offset pair include the more reliable center view will exceed 9,312 correct predictions while retaining the two-view design’s runtime and calibration gains.
change: Train 75% exact reflection pairs and 25% center-to-offset reflected pairs with symmetric consistency loss, and restore agreement-curved inference calibration.
mechanism: Center-anchored stochastic orbit consistency
evidence_used: Random cross-offset pairing retained 9,312 correct while lowering cross-entropy to 0.193848; the validated 1.5578 center weight indicates that anchoring those pairs to the center view should provide a more reliable consistency target.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing cross-offset reflected pairs from 25% to 50% will exceed 9,312 correct predictions by strengthening translation consistency while preserving reflection supervision and the successful two-view runtime profile.
change: Restore the best validated inference weighting and calibration, then train an equal mixture of exact-reflection and cross-offset-reflection pairs with the proven consistency weights.
mechanism: Balanced stochastic flip-offset orbit pairing
evidence_used: Moving from exact flip pairs to 25% cross-offset pairs retained 9,312 correct while reducing cross-entropy from 0.196252 to 0.193848; a balanced mixture tests whether the observed translation-robustness benefit continues at a stronger dose.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 25% classification loss on the averaged paired logits will exceed 9,312 correct predictions by directly optimizing the same logit-averaging behavior used during validation.
change: Replace separate per-view classification losses with an equivalent fused individual-view loss blended with cross-entropy on the pair’s mean logits; retain augmentation, consistency regularization, inference pooling, and runtime profile.
mechanism: Pair-ensemble aligned supervision
evidence_used: Stochastic flip-offset pairing retained 9,312 correct while reducing cross-entropy to 0.193848, whereas learned feature fusion fell to 9,280; this motivates supervising the successful output-level pair average without changing the architecture.
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
