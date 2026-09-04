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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 31.955160333076492, "validation_accuracy": 0.9111, "validation_correct": 9111, "validation_cross_entropy": 0.25698145141601564, "validation_score": 9111.397778343855}
prior_hypothesis: Extending the proven flip ensemble to five one-pixel translation views will exceed 9,094 correct predictions because training already exposes the model to random translations and horizontal reflections.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249466, "training_seconds": 60.428222542162985, "validation_accuracy": 0.9202, "validation_correct": 9202, "validation_cross_entropy": 0.23369549179077148, "validation_score": 9202.4052863963}
prior_hypothesis: Adding input-conditioned channel gating to the qualified multi-scale model will exceed 9,202 correct predictions by emphasizing class-relevant texture and silhouette channels while preserving the proven model exactly at initialization.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.06300445785746, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.23108364181518554, "validation_score": 9209.406146246296}
prior_hypothesis: Adding an inexpensive 7×7 spatial refinement block to the best multi-scale model will exceed 9,202 correct predictions by improving local feature interactions that global channel gating could not capture.

## Recent verification evidence

RECENT RESULT
hypothesis: Giving the centered flip pair twice the weight of each cardinally shifted pair will exceed 9,111 correct predictions by retaining the useful translation ensemble while reducing distortion from shifted views.
change: Keep the validated architecture, training recipe, and ten evaluation views, but replace uniform logit averaging with a 2:1 center-to-cardinal weighting.
mechanism: Center-weighted translation-and-flip logit ensemble
evidence_used: Cardinal translation averaging improved flip-only inference from 9,094 to 9,111 correct, while adding equally weighted diagonal translations reduced performance to 9,109, indicating that shifted views help but excessive uniform weighting can dilute the most reliable centered prediction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 30.49426345806569, "validation_accuracy": 0.9109, "validation_correct": 9109, "validation_cross_entropy": 0.25717334594726565, "validation_score": 9109.397717627096}

RECENT RESULT
hypothesis: Averaging calibrated class probabilities across the centered and four cardinal translation/flip pairs will exceed 9,111 correct predictions by limiting the influence of confidently incorrect shifted views.
change: Remove the harmful diagonal views, retain the best five-position ensemble, and replace arithmetic logit averaging with numerically stable probability averaging.
mechanism: Probability-space cardinal translation-and-flip ensemble
evidence_used: The five-position logit ensemble achieved 9,111 correct, while adding diagonal views reduced this to 9,109 and center weighting also reached only 9,109; this motivates changing how the proven ten views are combined rather than adding or manually weighting views.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 34.49561137496494, "validation_accuracy": 0.9112, "validation_correct": 9112, "validation_cross_entropy": 0.2599091430664062, "validation_score": 9112.396854013443}

RECENT RESULT
hypothesis: Averaging each original/flip pair in logit space before averaging the five position-level probabilities will exceed 9,112 correct predictions by preserving the proven flip consensus while isolating probability averaging to the less reliable translated views.
change: Keep the validated architecture and training recipe, but evaluate centered and four cardinal translations with hierarchical aggregation: logit averaging within each flip pair, then probability averaging across positions.
mechanism: Hierarchical flip-logit and translation-probability ensemble
evidence_used: Flip-logit averaging improved the residual model from 9,086 to 9,094 correct, while cardinal probability averaging reached the best result of 9,112; combining those successful aggregation rules according to augmentation type is the next direct test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240654, "training_seconds": 22.570345791056752, "validation_accuracy": 0.9112, "validation_correct": 9112, "validation_cross_entropy": 0.25859173583984374, "validation_score": 9112.397269412917}

RECENT RESULT
hypothesis: Replacing the sequential late CNN and location-specific flattening head with parallel standard/dilated receptive fields plus global mean/max statistics will exceed 9,112 correct predictions by learning complementary texture and silhouette representations.
change: Replace the classifier architecture with a 240,106-parameter residual multi-scale CNN while retaining the proven augmentation, optimization, and hierarchical test-time ensemble.
mechanism: Parallel multi-scale context fusion with global statistical pooling
evidence_used: Successive spatial-refinement changes improved correctness from 9,035 to 9,086, whereas extensive ensemble refinements plateaued at 9,112. This suggests representation is now the limiting factor. The old design assumes sequential 3×3 features and a flattened 7×7 head are sufficient; this patch instead learns parallel local/contextual features and translation-tolerant global statistics.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 240106, "training_seconds": 67.61626745807007, "validation_accuracy": 0.9202, "validation_correct": 9202, "validation_cross_entropy": 0.23267466125488281, "validation_score": 9202.405622031276}

RECENT RESULT
hypothesis: Adding input-conditioned channel gating to the qualified multi-scale model will exceed 9,202 correct predictions by emphasizing class-relevant texture and silhouette channels while preserving the proven model exactly at initialization.
change: Replace the current sequential CNN with Reference Design 3’s multi-scale global-statistics architecture, then add a 9,360-parameter squeeze gate initialized as an identity transformation; the resulting model has 249,466 learned parameters.
mechanism: Identity-initialized global channel recalibration
evidence_used: Reference Design 3 improved from 9,112 to 9,202 correct through multi-scale features and global mean/max pooling, while inference-ensemble refinements had plateaued. This motivates improving representation through adaptive channel selection without disturbing the qualified initial computation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249466, "training_seconds": 60.428222542162985, "validation_accuracy": 0.9202, "validation_correct": 9202, "validation_cross_entropy": 0.23369549179077148, "validation_score": 9202.4052863963}

RECENT RESULT
hypothesis: Training Reference Design 3 by minimizing cross-entropy on each image/flip pair’s mean logits will exceed 9,202 correct predictions because it directly optimizes the flip aggregation that previously improved validation correctness.
change: Restore the best 240,106-parameter multi-scale global-statistics architecture and train both horizontal orientations jointly through their averaged logits.
mechanism: Evaluation-matched paired-flip supervision
evidence_used: Reference Design 3 achieved the best tied accuracy and lowest cross-entropy at 9,202 correct, while flip-logit averaging previously improved the residual model from 9,086 to 9,094 correct; optimizing that same pairwise computation during training is the most direct untested extension.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding an inexpensive 7×7 spatial refinement block to the best multi-scale model will exceed 9,202 correct predictions by improving local feature interactions that global channel gating could not capture.
change: Add a residual depthwise/pointwise convolutional block after late multi-scale fusion and reduce the classifier width from 64 to 61, yielding 249,961 learned parameters.
mechanism: Late depthwise-separable spatial residual refinement
evidence_used: Multi-scale spatial representation raised correctness from 9,112 to 9,202, while spending the remaining capacity on channel gating tied at 9,202 with worse cross-entropy; this motivates using the capacity for additional spatial refinement instead.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249961, "training_seconds": 58.06300445785746, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.23108364181518554, "validation_score": 9209.406146246296}

RECENT RESULT
hypothesis: Extending the proven late refinement to two nonlinear 3×3 depthwise stages will exceed 9,209 correct predictions by modeling effective 5×5 spatial interactions while retaining more classifier capacity than a direct 5×5 kernel.
change: Replace channel gating with a two-stage depthwise/pointwise residual refinement block and reduce classifier width to 55, yielding approximately 249,799 parameters.
mechanism: Two-stage depthwise spatial refinement
evidence_used: Reference Design 3 reached 9,209 correct after spatial refinement, whereas channel gating tied the earlier 9,202 result with worse cross-entropy; this motivates spending the remaining capacity on deeper spatial processing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing single-location global maxima with the mean of each channel’s four strongest spatial responses will exceed 9,209 correct predictions by preserving salient features while reducing sensitivity to noisy or misaligned activation spikes.
change: Restore the verified 249,961-parameter multi-scale spatial-refinement model and its hierarchical test-time ensemble, but use top-four spatial averaging for the peak-statistics half of the classifier input.
mechanism: Robust top-k spatial peak pooling
evidence_used: Reference Design 3 achieved the best result—9,209 correct—using late spatial refinement and concatenated mean/max statistics; the large earlier gain from global-statistical pooling motivates refining its parameter-free peak statistic without adding the costly second refinement stage that timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing hard maxima with softmax-weighted peak statistics will exceed 9,209 correct predictions by retaining salient responses while distributing learning across several strong spatial locations.
change: Restore the verified 249,961-parameter spatial-refinement model and training recipe, but replace costly top-k pooling with efficient differentiable soft peak pooling.
mechanism: Smooth top-response spatial pooling
evidence_used: Spatial refinement achieved the best result at 9,209 correct, while robust top-four pooling was promising but timed out; smooth peak pooling tests the same robustness hypothesis without discrete top-k selection.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing depthwise spatial filtering followed by pointwise mixing with one grouped 3×3 convolution will exceed 9,209 correct predictions by learning spatial and cross-channel interactions jointly while preserving similar capacity and reducing sequential refinement operations.
change: Replace the late depthwise/pointwise refinement with a single 8-group 3×3 residual convolution and reduce classifier width from 61 to 60, yielding 249,854 learned parameters.
mechanism: Joint grouped spatial-channel residual refinement
evidence_used: Late spatial refinement produced the best result at 9,209 correct, whereas channel-only gating tied the earlier 9,202 result with worse cross-entropy; the proposed block strengthens the successful spatial mechanism without the extra stages that caused the two-stage design to time out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Expanding the verified late refinement from 3×3 to 5×5 will exceed 9,209 correct predictions by modeling broader spatial interactions without the sequential operations that caused the two-stage design to time out.
change: Restore the best spatial-refinement architecture, use a direct 5×5 depthwise kernel, and reduce classifier width to 53, yielding 249,873 learned parameters.
mechanism: Single-stage 5×5 depthwise spatial refinement
evidence_used: The 3×3 spatial-refinement model achieved the best result at 9,209 correct; a proposed two-stage effective-5×5 refinement timed out, motivating a computationally simpler single-stage test of the same broader-context hypothesis.
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
