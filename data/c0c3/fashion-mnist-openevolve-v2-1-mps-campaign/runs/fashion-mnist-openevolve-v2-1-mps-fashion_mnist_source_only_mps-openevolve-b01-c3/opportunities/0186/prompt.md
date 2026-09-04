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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 63.79398037493229, "validation_accuracy": 0.9324, "validation_correct": 9324, "validation_cross_entropy": 0.19407607765197754, "validation_score": 9324.418733788707}
prior_hypothesis: Retaining the validated 37.5% cross-offset rate while center-anchoring 71.37% of those pairs will exceed 9,325 correct predictions by matching training-view exposure to the inference ensemble’s measured 1.5578× center reliability.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.07845108397305, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1930408390045166, "validation_score": 9325.41909713704}
prior_hypothesis: Blending 12.5% reliability-weighted arithmetic probability pooling into the validated logit ensemble will exceed 9,325 correct predictions by damping isolated overconfident transformed views while preserving the stronger geometric-pooling signal.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.75497637502849, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19384812965393067, "validation_score": 9312.418813739856}
prior_hypothesis: Training 75% of pairs as exact reflections and 25% as reflected pairs from different validation offsets will exceed 9,312 correct predictions without the four-view design’s timeout.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 71.64032600005157, "validation_accuracy": 0.9289, "validation_correct": 9289, "validation_cross_entropy": 0.20669593772888184, "validation_score": 9289.414354589559}
prior_hypothesis: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Replacing the pooled-feature MLP with per-location class-evidence maps will exceed 9,325 correct predictions because it learns channel co-occurrences before spatial reduction instead of assuming independently pooled channels retain all discriminative structure.
change: Restore the validated 37.5% paired offset/flip training control, then replace global feature pooling followed by classification with a lightweight convolutional class head and learned per-class mean/max evidence pooling.
mechanism: Class-evidence mapping before spatial pooling
evidence_used: Reference Design 2 reached 9,325 correct with 37.5% paired offset exposure. The spatial-attention attempt timed out and only changed location weighting; this cheaper alternative tests the distinct load-bearing assumption that class interactions can occur after spatial information has already been discarded.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 25% reliability-weighted probability pooling into the validated logit ensemble will exceed 9,325 correct predictions by reducing the influence of extreme logits from unreliable transformed views while preserving the ensemble’s established center and flip-consistency weighting.
change: Keep training and compute unchanged; during evaluation, combine the existing weighted logit pool with a conservative weighted arithmetic probability pool before prediction and calibration.
mechanism: Hybrid logit–probability test-time aggregation
evidence_used: The current weighted ten-view logit ensemble achieved the best result of 9,325 correct, while center-anchored training fell to 9,324; this motivates refining inference aggregation without risking the repeated training-time failures caused by more elaborate training changes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring uniformly sampled 37.5% cross-offset pairs and assigning 12.5% of classification supervision to the pair-averaged logits will exceed 9,325 correct predictions by aligning training with the successful logit-averaged inference ensemble.
change: Remove the detrimental center anchoring, restore Reference Design 1’s cross-offset sampling, and blend individual-view cross-entropy with cross-entropy on each pair’s mean logits while retaining 5% consistency regularization.
mechanism: Pair-ensemble-aligned classification loss
evidence_used: Uniform 37.5% cross-offset training achieved the best verified result of 9,325 correct, while center anchoring reduced it to 9,324. The best design averages transformed-view logits at inference, motivating a conservative train-time objective on paired mean logits.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 76.32322312495671, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19431543159484863, "validation_score": 9312.418649869853}

RECENT RESULT
hypothesis: Using 37.5% cross-offset pairs while making those pairs differ only by translation will exceed 9,325 correct predictions by preserving the validated cross-offset dose while avoiding compounded flip-plus-offset disagreement.
change: Restore the best 37.5% cross-offset exposure and uniform 5% consistency, but use matching flip orientation for cross-offset pairs and opposite orientation for same-offset pairs.
mechanism: Disentangled translation and reflection pairing
evidence_used: Uniform 37.5% cross-offset training with 5% consistency achieved the best verified result of 9,325 correct; stronger translation exposure helped, while attempts to refine pair difficulty did not produce a completed result, motivating a compute-neutral isolation of translation consistency.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 77.25420095794834, "validation_accuracy": 0.9296, "validation_correct": 9296, "validation_cross_entropy": 0.1956116828918457, "validation_score": 9296.41819598048}

RECENT RESULT
hypothesis: Restoring the validated uniform 37.5% paired-offset training while increasing the inference penalty for flip-inconsistent offsets from 10% to 12.5% will exceed 9,325 correct predictions by suppressing unreliable transformed evidence without altering training compute.
change: Restore Reference Design 2’s paired-view loss and agreement-curved calibration, then strengthen only the flip-inconsistency downweighting.
mechanism: Slightly stronger flip-disagreement gating
evidence_used: Reference Design 2 achieved the best verified result of 9,325 correct using uniform 37.5% cross-offset pairs, 5% consistency, and a 10% flip-disagreement penalty; this patch preserves that control and isolates a conservative inference-only refinement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Blending 12.5% reliability-weighted arithmetic probability pooling into the validated logit ensemble will exceed 9,325 correct predictions by damping isolated overconfident transformed views while preserving the stronger geometric-pooling signal.
change: Keep training unchanged and replace evaluation logits with the log of an 87.5% logit-pooled and 12.5% probability-pooled mixture.
mechanism: Conservative hybrid geometric–arithmetic view pooling
evidence_used: The weighted logit ensemble produced the best verified 9,325 correct; the prior 25% probability-blend attempt timed out rather than yielding negative accuracy evidence, motivating a more conservative blend with negligible added computation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.07845108397305, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.1930408390045166, "validation_score": 9325.41909713704}



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
