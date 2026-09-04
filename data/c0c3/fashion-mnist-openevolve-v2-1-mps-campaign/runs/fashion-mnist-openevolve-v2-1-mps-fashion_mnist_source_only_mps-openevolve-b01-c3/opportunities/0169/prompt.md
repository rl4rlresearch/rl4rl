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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.6785941249691, "validation_accuracy": 0.9318, "validation_correct": 9318, "validation_cross_entropy": 0.19343022308349608, "validation_score": 9318.418960396953}
prior_hypothesis: Applying the proven 5% consistency strength to both exact-reflection and one-pixel cross-offset pairs will exceed 9,312 correct predictions by strengthening translation invariance without reducing exact-reflection pair frequency or increasing compute.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 78.51267020893283, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.1962520164489746, "validation_score": 9312.417972127214}
prior_hypothesis: Training every presented image alongside its horizontal mirror with a small symmetric consistency loss will reduce the flip instability that inference-time penalties identified, exceeding 9,289 correct predictions while retaining the best validated inference pooling and calibration.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.75497637502849, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19384812965393067, "validation_score": 9312.418813739856}
prior_hypothesis: Training 75% of pairs as exact reflections and 25% as reflected pairs from different validation offsets will exceed 9,312 correct predictions without the four-view design’s timeout.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.64032600005157, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669593772888184, "validation_score": 9289.414354589559}
prior_hypothesis: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Keeping 25% cross-offset reflected pairs but ramping their consistency weight from 0 to 5% will exceed 9,312 correct predictions by avoiding noisy early alignment while preserving the validated average 2.5% regularization.
change: Move stochastic offset construction into the loss, retain two supervised views per example, and cosine-ramp only the cross-offset consistency coefficient while exact-reflection pairs remain at 5%.
mechanism: Late-ramped cross-offset consistency
evidence_used: Static 25% cross-offset pairing retained 9,312 correct and reduced cross-entropy from 0.196252 to 0.193848; scheduling the same average cross-offset consistency is a compute-neutral test of whether early invariance pressure prevented a correctness gain.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling primary training offsets with the validated 1.5578× center weight while retaining the successful 75% exact-reflection/25% cross-offset pairing will exceed 9,312 correct predictions without increasing model compute.
change: Restore the best paired-consistency design and agreement-curved calibration, but importance-weight its primary offset sampling to match the center emphasis used by validation pooling.
mechanism: Center-weighted stochastic orbit consistency
evidence_used: Reference Design 3 achieved 9,312 correct and reduced cross-entropy to 0.193848 with uniform offset sampling, while its validated inference ensemble weights the center offset 1.5578×; aligning training exposure with that measured reliability is a compute-neutral next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying the proven 5% consistency strength to both exact-reflection and one-pixel cross-offset pairs will exceed 9,312 correct predictions by strengthening translation invariance without reducing exact-reflection pair frequency or increasing compute.
change: Restore the best validated paired-consistency design and inference calibration, while increasing cross-offset consistency from 2.5% to 5%.
mechanism: Equal-strength flip-offset orbit consistency
evidence_used: Introducing 25% cross-offset pairs at 2.5% consistency retained 9,312 correct and reduced cross-entropy from 0.196252 to 0.193848; equal-strength consistency directly tests whether the weaker translation constraint limited correctness.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.6785941249691, "validation_accuracy": 0.9318, "validation_correct": 9318, "validation_cross_entropy": 0.19343022308349608, "validation_score": 9318.418960396953}



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
