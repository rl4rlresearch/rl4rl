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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 78.42631683289073, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.21967102966308594, "validation_score": 9281.409946606782}
prior_hypothesis: Uniformly averaging two of every three iterates across the final 15% of training will exceed 9,282 correct predictions by capturing broader low-learning-rate trajectory diversity while performing the same 80 averaging updates as the proven final-10% average.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 62.886122666997835, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.2185217487335205, "validation_score": 9275.410333258737}
prior_hypothesis: Averaging only the final 5% of iterates will exceed 9,282 correct predictions by retaining the proven variance reduction of tail averaging while reducing mismatch with terminal BatchNorm statistics and halving averaging overhead.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 59.95856470800936, "validation_accuracy": 0.9285, "validation_correct": 9285, "validation_cross_entropy": 0.21857940216064453, "validation_score": 9285.41031384505}
prior_hypothesis: Linearly weighting later iterates within the proven final-10% averaging window will exceed 9,282 correct predictions by preserving its beneficial temporal coverage while making the installed weights more consistent with terminal BatchNorm statistics.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.06300445785746, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.23108364181518554, "validation_score": 9209.406146246296}
prior_hypothesis: Adding an inexpensive 7×7 spatial refinement block to the best multi-scale model will exceed 9,202 correct predictions by improving local feature interactions that global channel gating could not capture.

## Recent verification evidence

RECENT RESULT
hypothesis: Preserving 2×2 feature layout alongside global mean and peak statistics will exceed 9,285 correct predictions because it exposes class-relevant spatial arrangement that the shared globally invariant readout discards.
change: Replace the global-only classifier input with a 2×2 spatial pyramid plus global summaries, resize its bottleneck to remain under 250,000 parameters, and evaluate it using the proven final-10% recency-weighted averaging procedure.
mechanism: Coarse spatial-pyramid classification head
evidence_used: Local spatial refinement improved correctness from 9,202 to 9,209, and position-matched augmentation reached 9,262, indicating that spatial structure matters; however, all available designs still erase feature layout with mean/max pooling. The 9,285-result establishes recency-weighted averaging as the strongest training baseline for testing this alternative representation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing global mean pooling with per-channel learned 7×7 spatial attention will exceed 9,285 correct predictions by preserving class-relevant feature layout, while linear final-10% averaging retains the strongest verified optimization baseline.
change: Add zero-initialized separable row/column attention for each late feature channel, reduce the classifier width to remain below 250,000 parameters, and restore the verified final-10% linearly recency-weighted parameter average.
mechanism: Channel-specific separable spatial attention pooling
evidence_used: Final-10% linear recency weighting achieved the best result at 9,285 correct; spatial refinement and evaluation-matched translations showed that spatial structure matters, yet the current global mean statistic discards feature location entirely.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249884, "training_seconds": 77.96925141708925, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.22021239547729493, "validation_score": 9257.409764727725}

RECENT RESULT
hypothesis: Reducing BatchNorm momentum to 0.04 during final-10% linear parameter averaging will exceed 9,285 correct predictions by aligning normalization statistics with the average model’s effective trajectory position without extra forward passes.
change: Restore cardinal-shift augmentation and final-10% linear recency-weighted averaging, while slowing BatchNorm running-stat updates during the averaging window.
mechanism: Tail-matched BatchNorm momentum with recency-weighted parameter averaging
evidence_used: Linear recency-weighted averaging achieved the best result at 9,285 correct, while explicit buffer averaging and recalibration timed out; tail-specific BatchNorm momentum targets the same normalization mismatch with negligible overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Quadratically weighting later iterates within the final 10% window will exceed 9,285 correct predictions by further reducing BatchNorm-state mismatch while preserving the full successful averaging span.
change: Replace linear recency weights with exact online weights proportional to the square of each tail iterate’s index.
mechanism: Quadratic-recency tail parameter averaging
evidence_used: Final-10% linear weighting achieved 9,285 correct versus 9,282 for uniform weighting, showing that recency bias improved the strongest baseline without extra training work; quadratic weighting directly tests whether a stronger smooth bias extends that gain.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Giving the native centered view half of the inference probability mass while restoring final-10% linear recency averaging will exceed 9,285 correct predictions by retaining translation robustness without letting four shifted crops overwhelm the validation image’s true alignment.
change: Restore the strongest verified final-10% linearly weighted parameter average and change inference from uniform position averaging to 50% centered and 12.5% per cardinal shift.
mechanism: Center-weighted translation ensemble with linear tail averaging
evidence_used: Final-10% linear recency averaging achieved the best verified result of 9,285 correct; cardinal translations were beneficial, while the unverified center-biased training proposal identified native-alignment dilution as a plausible remaining issue. Weighting only inference isolates that idea without additional training cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting final-10% iterates proportional to their index^1.5 will exceed 9,285 correct predictions by increasing the beneficial recency bias of linear averaging while retaining the temporal coverage lost by final-5% averaging.
change: Restore the proven final-10% averaging window and replace uniform averaging with exact online 1.5-power recency weighting.
mechanism: Intermediate-recency tail parameter averaging
evidence_used: Final-10% linear weighting achieved 9,285 correct versus 9,282 for uniform weighting, while final-5% averaging fell to 9,275; this motivates testing a stronger intermediate recency profile without shortening the successful window.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing label smoothing to zero before the proven final-10% linear tail average will exceed 9,285 correct predictions by retaining early regularization while sharpening terminal decision boundaries.
change: Restore cardinal translation training and linear recency-weighted averaging, while holding 0.02 label smoothing through 75% of training and reducing it to zero by the averaging window.
mechanism: Hard-target tail refinement with recency-weighted averaging
evidence_used: Final-10% linear averaging achieved the best verified result at 9,285 correct, outperforming uniform averaging and the shorter 5% window; this motivates refining the terminal trajectory without shortening its successful averaging span.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Leaving BatchNorm scale and bias at their terminal values while linearly averaging all other parameters will exceed 9,285 correct predictions by better matching the terminal running statistics.
change: Preserve the proven final-10% recency-weighted average, but exclude BatchNorm trainable parameters from it.
mechanism: Terminal BatchNorm affine anchoring during tail averaging
evidence_used: Linear recency weighting improved uniform final-10% averaging from 9,282 to 9,285 correct, supporting reduced mismatch with terminal BatchNorm state; this targets that mismatch without extra passes or averaging work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Quadratically weighting the sampled final-15% iterates will exceed 9,285 correct predictions by retaining broader trajectory diversity while shifting the average model’s effective time toward that of the successful final-10% linear average.
change: Keep the current 80-update subsampled final-15% window, but replace uniform averaging with exact online weights proportional to each sampled iterate index squared.
mechanism: Moment-matched extended-tail averaging
evidence_used: Final-10% linear recency weighting achieved 9,285 correct versus 9,282 for uniform weighting, while the subsampled final-15% uniform average reached 9,281; quadratic weighting directly adds the proven recency preference to the broader window without increasing averaging work.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Uniformly averaging probabilities across all ten independently sampled training transformations will exceed 9,285 correct predictions by avoiding premature flip-logit fusion, while retaining the strongest verified final-10% linear recency average.
change: Restore final-10% linearly weighted parameter averaging and replace hierarchical inference aggregation with a uniform probability mixture over every cardinal-shift and flip view.
mechanism: Transformation-matched probability ensembling with linear tail averaging
evidence_used: Final-10% linear recency averaging achieved the best verified result of 9,285 correct. Training samples uniformly from ten position/flip transformations, but current inference first geometrically combines flip pairs; directly marginalizing all ten view probabilities better matches that training distribution without additional inference work.
result: training did not finish within the verification time limit

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
