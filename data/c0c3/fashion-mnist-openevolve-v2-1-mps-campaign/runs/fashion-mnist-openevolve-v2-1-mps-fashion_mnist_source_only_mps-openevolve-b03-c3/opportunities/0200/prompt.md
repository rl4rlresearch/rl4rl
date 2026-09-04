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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 49.4281909998972, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725957565307618, "validation_score": 9284.414161138237}
prior_hypothesis: An evaluation scale of 1.1645 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207286140.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.361905792029575, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20725854110717773, "validation_score": 9284.414161493147}
prior_hypothesis: An evaluation scale of 1.16727 will preserve all 9,284 predictions while reducing cross-entropy below 0.207258652.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 36.21409408305772, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072585422515869, "validation_score": 9284.414161492756}
prior_hypothesis: An evaluation scale of 1.167286 will preserve all 9,284 argmax predictions while reducing validation cross-entropy below 0.207258541.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 44.8443427500315, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.2072586524963379, "validation_score": 9284.414161454933}
prior_hypothesis: An evaluation scale of 1.1682 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207259576.

## Recent verification evidence

RECENT RESULT
hypothesis: Setting the evaluation scale to 1.16727 will preserve 9,284 correct predictions while reducing validation cross-entropy below the current 0.207259576.
change: Replace only the evaluation logit multiplier with the best verified scale.
mechanism: Best empirically validated global logit calibration
evidence_used: Reference Design 1 achieved the highest available validation_score, 9284.414161493147, at scale 1.16727; the nearby 1.167286 scale produced marginally higher cross-entropy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing the batch size to 128 will exceed 9,284 correct predictions by increasing optimizer updates from 522 to approximately 782 within the fixed 100,000-example budget, while the paired-image loss still gives BatchNorm an effective 256-image batch.
change: Change only the training batch size from 192 to 128, preserving the validated architecture, loss, augmentation, TTA, and calibration.
mechanism: Increased finite-exposure optimization granularity
evidence_used: Calibration is saturated at 9,284 correct, while channel attention, aligned translation sampling, probability-space TTA, and late EMA all reduced accuracy; batch granularity remains an untested way to improve convergence without altering the established classifier computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245152, "training_seconds": 44.35637570801191, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.2046711456298828, "validation_score": 9264.415051030162}

RECENT RESULT
hypothesis: Adding content-dependent pairwise interactions across the 7×7 feature map will exceed 9,284 correct predictions by modeling relationships between distant image parts that the local refinement stack cannot represent directly.
change: Preserve the validated convolutional, aggregation, classifier, training, and TTA paths while adding a 3,841-parameter low-rank spatial-attention residual before aggregation; total learned parameters become 248,993.
mechanism: Low-rank nonlocal spatial-relation residual
evidence_used: Mean–max channel attention fell to 9,274 correct, indicating adaptive channel importance was not the missing mechanism. This patch instead challenges the shared assumption that local convolutions alone sufficiently relate separated shape parts, while retaining the established layout-sensitive decision path whose replacement previously regressed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Learning an independent max-versus-average blend for each feature channel will exceed 9,284 correct predictions by preserving the validated initial computation while allowing edge, texture, and shape channels to pool differently.
change: Replace each scalar pooling mixture with a broadcast channel-wise mixture, initialize it to the current scalar value, and use the best verified evaluation scale of 1.16727.
mechanism: Channel-specific mixed pooling
evidence_used: Content-dependent channel attention reduced accuracy to 9,274, motivating a simpler 70-parameter static channel specialization; Reference Design 2 established 1.16727 as the best verified calibration.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245222, "training_seconds": 41.42501541599631, "validation_accuracy": 0.9282, "validation_correct": 9282, "validation_cross_entropy": 0.2073271797180176, "validation_score": 9282.41413794736}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.16727 will preserve all 9,284 argmax predictions while lowering cross-entropy below the current 0.207259576.
change: Replace only the evaluation-logit multiplier with the strongest verified calibration.
mechanism: Best-validated global logit calibration
evidence_used: Reference Design 1 achieved the highest available validation_score, 9284.414161493147, with 9,284 correct and cross-entropy 0.207258541; positive scaling cannot change argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A compute-efficient nonlinear channel-mixing residual will exceed 9,284 correct predictions while retaining the validated model’s initial computation and staying below the parameter ceiling.
change: Add a zero-initialized 80→28→80 pointwise residual before aggregation, increasing learned parameters from 245,152 to 249,848 while preserving the existing training and evaluation paths.
mechanism: Zero-initialized pointwise channel-refinement residual
evidence_used: Channel reweighting and channel-specific pooling regressed, while spatial attention exceeded the verification time limit; this tests inexpensive nonlinear feature refinement rather than attention or pooling changes, with zero initialization protecting the established baseline.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249848, "training_seconds": 38.60794583382085, "validation_accuracy": 0.923, "validation_correct": 9230, "validation_cross_entropy": 0.2110298553466797, "validation_score": 9230.412871737053}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.16727 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207258542.
change: Replace only the evaluation-logit multiplier with the strongest verified scale.
mechanism: Best-validated global logit calibration
evidence_used: Reference Design 2 achieved the highest available validation_score at scale 1.16727 with 9,284 correct and cross-entropy 0.207258541; the current 1.167286 scale produced slightly higher cross-entropy, while positive global scaling cannot change class rankings.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding independent ±8° rotations after the existing translations will increase validation_correct above 9,284 by learning small orientation robustness not covered by prior augmentation, architecture, or ensemble experiments.
change: Apply bilinear, border-padded random rotations during training and use the best verified evaluation scale of 1.16727.
mechanism: Mild rotational augmentation
evidence_used: Translation-distribution alignment fell to 9,262 correct and several architectural refinements regressed, while calibration has saturated at 9,284; this motivates testing a distinct label-preserving augmentation while retaining the validated model and TTA.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Explicitly reducing disagreement between paired original and flipped predictions late in training will exceed 9,284 correct predictions while preserving the validated logit-space TTA.
change: Add a symmetric detached-target KL loss between existing paired logits, ramp its weight from zero to 0.20, and use the best verified evaluation scale of 1.16727.
mechanism: Late-ramped symmetric flip-consistency regularization
evidence_used: Arithmetic probability TTA reduced accuracy to 9,280, so the validated logit ensemble should remain intact; the existing paired training path supplies both views but does not directly penalize their disagreement, enabling this consistency test without architectural or computational expansion.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 42.769176999805495, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.20789540328979492, "validation_score": 9277.41394312673}

RECENT RESULT
hypothesis: Decaying dropout from 0.10 to zero will exceed 9,284 correct predictions by retaining early regularization while making low-learning-rate fine-tuning match deterministic evaluation.
change: Preserve the validated architecture, loss, augmentation, optimizer, TTA, and calibration while annealing the existing classifier dropout probability over training.
mechanism: Cosine-annealed classifier dropout
evidence_used: Architectural changes, smaller batches, EMA, and flip-consistency regularization all reduced validation_correct; this motivates a minimal schedule change that removes only late training-time noise from the otherwise validated computation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 47.81579783302732, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.207254407119751, "validation_score": 9276.414162911356}

RECENT RESULT
hypothesis: Adding a Laplacian image-basis channel will exceed 9,284 correct predictions by exposing signed stroke curvature and boundary transitions that the existing first-order gradient channels do not represent directly.
change: Add a scaled four-neighbor Laplacian as a sixth input channel, expand the first convolution accordingly, and use the best verified evaluation calibration.
mechanism: Explicit second-order curvature feature
evidence_used: Post-feature architectural additions consistently reduced accuracy, while the established handcrafted image basis remains untested for second-order information; the 1.16727 scale achieved the best verified cross-entropy at 9,284 correct.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245368, "training_seconds": 36.2046590000391, "validation_accuracy": 0.9248, "validation_correct": 9248, "validation_cross_entropy": 0.2088061782836914, "validation_score": 9248.413631241288}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.16727 will preserve all 9,284 argmax predictions while reducing validation cross-entropy below 0.207258652.
change: Replace only the evaluation-logit multiplier with the strongest verified scale.
mechanism: Best-validated global logit calibration
evidence_used: Reference Design 2 achieved the highest reported validation_score, 9284.414161493147, with 9,284 correct and cross-entropy 0.207258541; positive global scaling cannot change argmax predictions.
result: the implementation could not be verified

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
