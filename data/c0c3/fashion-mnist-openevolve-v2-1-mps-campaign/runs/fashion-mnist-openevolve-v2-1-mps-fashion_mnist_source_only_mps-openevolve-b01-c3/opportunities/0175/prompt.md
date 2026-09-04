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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.36925508314744, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19312731437683106, "validation_score": 9325.419066761757}
prior_hypothesis: Increasing cross-offset reflected pairs from 25% to 37.5% while retaining 5% consistency will exceed 9,318 correct predictions by strengthening the beneficial translation constraint without moving fully to the unverified 50% mixture.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.75497637502849, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19384812965393067, "validation_score": 9312.418813739856}
prior_hypothesis: Training 75% of pairs as exact reflections and 25% as reflected pairs from different validation offsets will exceed 9,312 correct predictions without the four-view design’s timeout.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.64032600005157, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669593772888184, "validation_score": 9289.414354589559}
prior_hypothesis: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Raising only cross-offset consistency from 5% to 7.5% while preserving 5% exact-reflection consistency will exceed 9,318 correct predictions by extending the observed benefit of stronger translation alignment.
change: Apply per-example consistency weights of 0.075 to cross-offset pairs and 0.05 to exact-reflection pairs.
mechanism: Selectively stronger cross-offset consistency
evidence_used: At the same 25% cross-offset frequency, increasing cross-offset consistency from 2.5% produced 9,312 correct to 5% producing 9,318, while exact-reflection consistency remained successful at 5%; a selective intermediate increase tests whether that positive dose response continues without weakening reflection training.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing fixed global-average pooling with a zero-initialized learned spatial attention pool, while using the verified 5% consistency strength for every pair, will exceed 9,318 correct predictions by learning which image regions should contribute to each representation.
change: Preserve the successful backbone, max-pooled evidence, paired training, and inference ensemble; replace uniform mean aggregation with learned attention initialized to identical behavior, slightly narrow the classifier to remain below 250,000 parameters, and restore equal-strength cross-offset consistency.
mechanism: Content-adaptive saliency pooling
evidence_used: Equal 5% flip-offset consistency produced the best result at 9,318 correct, while further orbit-loss variations timed out and learned reflection fusion fell to 9,280. Those designs retain or manipulate fixed spatial aggregation; this patch instead challenges the load-bearing assumption that every spatial location should contribute equally to the global feature vector.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Excluding opposite two-pixel offset pairs while retaining 25% cross-offset sampling and 5% consistency will exceed the best result of 9,318 correct predictions.
change: Restore the best stochastic paired-view design, but restrict cross-offset partners to transformations differing by at most one pixel per axis.
mechanism: Bounded-displacement flip-offset consistency
evidence_used: Equal-strength cross-offset consistency improved correctness from 9,312 to 9,318; its unrestricted partner sampling includes noisier opposite offsets separated by two pixels, so removing only those outliers tests a cleaner translation constraint without extra model compute.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising only cross-offset consistency from the validated 5% to 6.25% will exceed 9,318 correct predictions while preserving the successful exact-reflection constraint and runtime profile.
change: Restore stochastic paired-view training with 25% cross-offset pairs, retain 5% exact-reflection consistency, apply 6.25% consistency to cross-offset pairs, and restore agreement-curved inference calibration.
mechanism: Midpoint-strength cross-offset consistency
evidence_used: Increasing cross-offset consistency from 2.5% to 5% improved correctness from 9,312 to 9,318; 6.25% is a conservative midpoint toward the unverified 7.5% attempt without changing compute.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing cross-offset reflected pairs from 25% to 37.5% while retaining 5% consistency will exceed 9,318 correct predictions by strengthening the beneficial translation constraint without moving fully to the unverified 50% mixture.
change: Raise the stochastic cross-offset pair frequency to 37.5%; preserve the architecture, two-view compute, consistency strength, and inference ensemble.
mechanism: Moderately increased flip-offset orbit exposure
evidence_used: Raising cross-offset consistency from 2.5% to 5% improved correctness from 9,312 to 9,318, showing that stronger translation alignment helps; 37.5% is a conservative midpoint between the validated 25% frequency and the 50% attempt that did not finish.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.36925508314744, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19312731437683106, "validation_score": 9325.419066761757}

RECENT RESULT
hypothesis: Increasing cross-offset pair frequency from the validated 37.5% to 43.75% while retaining uniform 5% consistency will exceed 9,325 correct predictions without changing model compute.
change: Use cross-offset reflected partners for 43.75% of examples and apply the validated 5% consistency weight to every pair.
mechanism: Intermediate-strength flip-offset orbit exposure
evidence_used: Raising cross-offset frequency from 25% to 37.5% at 5% consistency improved correctness from 9,318 to 9,325; 43.75% is a conservative midpoint toward the 50% attempt that timed out despite unchanged tensor shapes.
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
