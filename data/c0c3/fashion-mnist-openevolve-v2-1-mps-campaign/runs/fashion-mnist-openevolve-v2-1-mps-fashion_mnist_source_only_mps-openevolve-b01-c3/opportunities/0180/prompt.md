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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.64032600005157, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669593772888184, "validation_score": 9289.414354589559}
prior_hypothesis: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 63.79398037493229, "validation_accuracy": 0.9324, "validation_correct": 9324, "validation_cross_entropy": 0.19407607765197754, "validation_score": 9324.418733788707}
prior_hypothesis: Retaining the validated 37.5% cross-offset rate while center-anchoring 71.37% of those pairs will exceed 9,325 correct predictions by matching training-view exposure to the inference ensemble’s measured 1.5578× center reliability.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 75.36925508314744, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.19312731437683106, "validation_score": 9325.419066761757}
prior_hypothesis: Increasing cross-offset reflected pairs from 25% to 37.5% while retaining 5% consistency will exceed 9,318 correct predictions by strengthening the beneficial translation constraint without moving fully to the unverified 50% mixture.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.75497637502849, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19384812965393067, "validation_score": 9312.418813739856}
prior_hypothesis: Training 75% of pairs as exact reflections and 25% as reflected pairs from different validation offsets will exceed 9,312 correct predictions without the four-view design’s timeout.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Using exactly 43.75% cross-offset pairs in every minibatch will exceed 9,325 correct predictions by extending the validated 25%-to-37.5% improvement while reducing minibatch composition variance.
change: Restore paired offset training, assign a rotating 7-of-16 examples to cross-offset reflected partners, and retain uniform 5% symmetric consistency regularization.
mechanism: Stratified 7-of-16 flip-offset orbit exposure
evidence_used: Increasing cross-offset frequency from 25% to 37.5% at 5% consistency improved validation correctness from 9,318 to 9,325. The randomized 43.75% attempt timed out, leaving its accuracy hypothesis unresolved; stratification tests that dose with identical model compute and a fixed pair proportion.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising cross-offset reflected pairs from the validated 37.5% to 40.625% while retaining uniform 5% consistency will exceed 9,325 correct predictions without changing tensor shapes or model compute.
change: Restore the best validated paired-view training and agreement-curved inference calibration, then increase cross-offset pair frequency to 40.625%.
mechanism: Conservative flip-offset orbit dose refinement
evidence_used: Increasing cross-offset frequency from 25% to 37.5% at 5% consistency improved validation correctness from 9,318 to 9,325; 40.625% is a conservative interpolation toward the unresolved 43.75% dose.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Ramping cross-offset pairs from 25% to 50% while preserving the validated 37.5% average exposure will exceed 9,325 correct predictions by emphasizing exact-reflection learning early and translation invariance after class features stabilize.
change: Replace the fixed 37.5% cross-offset probability with a linear 25%-to-50% training schedule; preserve architecture, loss strength, compute, and inference pooling.
mechanism: Late-ramped flip-offset orbit exposure
evidence_used: Raising fixed cross-offset exposure from 25% to 37.5% improved validation correctness from 9,318 to 9,325, establishing a beneficial dose response; this tests whether scheduling the same average dose improves when that invariance is learned.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 77.43810262507759, "validation_accuracy": 0.9299, "validation_correct": 9299, "validation_cross_entropy": 0.19667639923095703, "validation_score": 9299.417823899863}

RECENT RESULT
hypothesis: Combining the validated 37.5% cross-offset exposure with a modest 5.625% cross-offset consistency weight will exceed 9,325 correct predictions while preserving 5% exact-reflection regularization.
change: Increase cross-offset pair frequency from 25% to 37.5% and selectively raise only their consistency weight from 2.5% to 5.625%.
mechanism: Conservative joint orbit-regularization refinement
evidence_used: At 25% exposure, raising cross-offset consistency from 2.5% to 5% improved correctness from 9,312 to 9,318; raising exposure to 37.5% at 5% further improved it to 9,325. This tests a conservative continuation of both validated dose responses.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Retaining the validated 37.5% cross-offset rate while center-anchoring 71.37% of those pairs will exceed 9,325 correct predictions by matching training-view exposure to the inference ensemble’s measured 1.5578× center reliability.
change: Restore the best paired-view training design, but construct a calibrated mixture of random and center-to-side cross-offset pairs whose marginal offset frequencies match validation pooling weights.
mechanism: Validation-weight-matched center anchoring
evidence_used: Reference Design 1 achieved the best result—9,325 correct—at 37.5% cross-offset exposure, and its validated inference ensemble weights the center offset 1.5578× more than each side offset; this patch preserves the successful dose while aligning training exposure with that reliability signal.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 63.79398037493229, "validation_accuracy": 0.9324, "validation_correct": 9324, "validation_cross_entropy": 0.19407607765197754, "validation_score": 9324.418733788707}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
