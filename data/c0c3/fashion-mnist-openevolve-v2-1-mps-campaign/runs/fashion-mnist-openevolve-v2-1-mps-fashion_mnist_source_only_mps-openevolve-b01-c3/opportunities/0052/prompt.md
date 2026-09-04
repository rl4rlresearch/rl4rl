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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.06300445785746, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.23108364181518554, "validation_score": 9209.406146246296}
prior_hypothesis: Adding an inexpensive 7×7 spatial refinement block to the best multi-scale model will exceed 9,202 correct predictions by improving local feature interactions that global channel gating could not capture.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.09104441688396, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.21856914138793945, "validation_score": 9283.410317300035}
prior_hypothesis: Linearly averaging every other iterate across the final 10% will exceed 9,281 correct predictions by recovering more of the dense average’s 9,285-correct benefit while retaining substantially lower averaging overhead.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.99758079089224, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.21854879875183106, "validation_score": 9281.410324149932}
prior_hypothesis: Linearly averaging 20 evenly spaced iterates across the final 10% will exceed the current 9,281 correct predictions while reducing averaging overhead enough to finish verification.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.95856470800936, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.21857940216064453, "validation_score": 9285.41031384505}
prior_hypothesis: Linearly weighting later iterates within the proven final-10% averaging window will exceed 9,282 correct predictions by preserving its beneficial temporal coverage while making the installed weights more consistent with terminal BatchNorm statistics.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing fixed-kernel late refinement with lightweight four-head self-attention will exceed 9,285 correct predictions by learning image-dependent relationships between distant garment parts before global pooling.
change: Replace the 7×7 depthwise refinement block with relative-position-aware spatial self-attention, while restoring cardinal-shift augmentation and the strongest verified final-10% linear recency-weighted averaging procedure.
mechanism: Content-adaptive nonlocal spatial relation block
evidence_used: Local refinement improved correctness from 9,202 to 9,209, showing value in late spatial interaction, while static attention pooling reached only 9,257; Reference Design 3 reached 9,285 with linear tail averaging. This challenges the load-bearing assumption that fixed local kernels provide sufficient spatial reasoning, using dynamic all-pairs feature interaction rather than another pooling modification.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sampling and ensembling the centered view with 50% probability will exceed 9,285 correct predictions by preserving translation robustness while reducing dilution of the validation images’ native alignment.
change: Give the centered crop half the training and inference probability mass, with the four cardinal shifts receiving one eighth each; retain the proven architecture and final-10% linear recency average.
mechanism: Native-alignment-matched translation marginalization
evidence_used: The strongest verified design reached 9,285 correct with cardinal translations and linear tail averaging, while the inference-only center-weighting attempt timed out without accuracy evidence; matching both training and inference distributions tests the native-alignment hypothesis without additional computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified final-10% linear tail average will exceed the current 9,281 correct predictions and recover approximately 9,285 correct.
change: Replace the subsampled uniform final-15% average with averaging every final-10% iterate using exact linear recency weights.
mechanism: Final-10% linearly recency-weighted parameter averaging
evidence_used: Reference Design 2 achieved the strongest verified result—9,285 correct—while the current broader subsampled uniform average achieved 9,281.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Exactly balancing the ten position/flip transformations within each minibatch will exceed 9,285 correct predictions by reducing augmentation-gradient variance, while the verified final-10% linear average preserves the strongest optimization baseline.
change: Restore final-10% linearly recency-weighted parameter averaging and replace independent random augmentation draws with a deterministic, step-rotated balance across all ten training transformations.
mechanism: Stratified transformation sampling with linear tail averaging
evidence_used: Reference Design 2 achieved the best verified result of 9,285 correct using final-10% linear averaging and uniform transformation sampling; the current final-5% average fell to 9,275, while stratification makes each limited-exposure minibatch more closely match the uniform ten-view inference distribution without added model computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the 9,285-correct cardinal-shift design while fusing its final-10% parameter averaging will recover at least 9,285 correct predictions and finish within the verification time limit.
change: Restore five-way cardinal translation augmentation and exact final-10% linear recency averaging, using foreach tensor operations to reduce averaging overhead.
mechanism: Fused linear-recency tail averaging
evidence_used: Reference Design 3 achieved the best verified result of 9,285 correct in 59.96 seconds; later implementations repeatedly timed out, making a computationally fused reproduction of that proven method the most informative next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Setting BatchNorm momentum to 0.04 will exceed 9,285 correct predictions by aligning running statistics with the effective temporal center of the successful final-10% linear parameter average.
change: Set every BatchNorm2d layer’s running-stat momentum to 0.04 without changing training gradients, parameter averaging, or per-step computation.
mechanism: Tail-average-aligned BatchNorm statistics
evidence_used: Linear final-10% weighting improved uniform averaging from 9,282 to 9,285 correct, suggesting better temporal alignment with terminal BatchNorm statistics; momentum 0.04 shifts their effective center toward the linearly averaged parameters while preserving the beneficial full window.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Linearly averaging 20 evenly spaced iterates across the final 10% will exceed the current 9,281 correct predictions while reducing averaging overhead enough to finish verification.
change: Restore the strongest final-10% linear-recency averaging distribution, but sample every fourth iterate so its effective temporal center is preserved with one quarter as many averaging updates.
mechanism: Temporally sparse linear tail averaging
evidence_used: Final-10% linear averaging achieved the best verified result of 9,285 correct, whereas the current final-15% sparse uniform average reached 9,281; repeated full-density restorations timed out, motivating a compute-reduced approximation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.99758079089224, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.21854879875183106, "validation_score": 9281.410324149932}

RECENT RESULT
hypothesis: Linearly averaging every other iterate across the final 10% will exceed 9,281 correct predictions by recovering more of the dense average’s 9,285-correct benefit while retaining substantially lower averaging overhead.
change: Extend averaging from the final 5% to the final 10%, sample every second iterate, and weight sampled iterates linearly by recency.
mechanism: Half-density linear-recency tail averaging
evidence_used: Dense final-10% linear averaging achieved 9,285 correct, while every-fourth-iterate averaging achieved 9,281 and completed verification; half-density sampling directly tests the accuracy–runtime midpoint.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 66.09104441688396, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.21856914138793945, "validation_score": 9283.410317300035}

RECENT RESULT
hypothesis: Restoring the verified dense final-10% linear average will recover 9,285 correct predictions, while a 0.90 inference temperature will preserve every argmax and reduce validation cross-entropy below 0.218579.
change: Restore cardinal-shift augmentation and dense linear-recency parameter averaging from Reference Design 3, then sharpen only the aggregated evaluation logits.
mechanism: Temperature-calibrated linear tail averaging
evidence_used: Reference Design 3 achieved the best verified result at 9,285 correct; its 0.02 label smoothing and ten-view probability aggregation motivate testing mild confidence sharpening without changing predicted classes.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Linearly averaging BatchNorm running statistics over the same final-10% window as learned parameters will exceed 9,285 correct predictions by eliminating the mismatch between averaged weights and terminal normalization statistics.
change: Track floating-point model buffers and apply the existing linear-recency tail average to them, adding negligible computation and no learned parameters.
mechanism: Tail-aligned BatchNorm buffer averaging
evidence_used: Linear-recency parameter averaging achieved the best verified result of 9,285 correct versus 9,282 for uniform averaging; the current design still pairs those averaged parameters with terminal BatchNorm statistics, making normalization-state alignment the most targeted next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing purely translation-invariant global pooling with learned channel projections summarized as row and column activation profiles will exceed 9,285 correct predictions by exposing garment-part arrangement while retaining robust global mean and peak features.
change: Challenge the assumption that global mean/max statistics contain sufficient shape information: project late features into 16 layout channels, preserve their seven-row and seven-column profiles, and classify the combined global-and-layout representation through a parameter-budgeted head.
mechanism: Learned axial-profile classification head
evidence_used: The strongest global-pooling design reached 9,285 correct, while changing its averaging density moved accuracy by only 2–4 images and repeated averaging variants timed out. Local spatial refinement improved correctness from 9,202 to 9,209, indicating spatial organization is useful; axial profiles test that signal directly without costly all-pairs attention and remain under 250,000 parameters.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Densely averaging only the classifier while averaging the backbone every fourth iterate will exceed 9,283 correct predictions by recovering decision-boundary stability from the 9,285-correct dense average without its full runtime cost.
change: Keep the proven sparse final-10% average for all parameters, add a separate dense linear-recency average for the small classifier, and install the dense classifier average after the sparse full-model average.
mechanism: Hybrid-density linear tail averaging
evidence_used: Increasing final-10% averaging density improved correctness monotonically from 9,281 every fourth iterate, to 9,283 every second iterate, to 9,285 every iterate; concentrating dense averaging on the classifier targets prediction-changing parameters with much less overhead.
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
