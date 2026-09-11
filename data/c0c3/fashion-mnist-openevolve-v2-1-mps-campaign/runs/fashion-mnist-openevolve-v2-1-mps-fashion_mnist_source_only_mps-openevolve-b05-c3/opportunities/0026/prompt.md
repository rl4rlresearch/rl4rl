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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.43390987510793, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.24108177490234375, "validation_score": 9163.402874339235}
prior_hypothesis: Beginning center/cardinal one-pixel augmentation after five-eighths of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the timed-out two-fifths configuration.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.46581783308648, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.23885296478271484, "validation_score": 9167.403599147125}
prior_hypothesis: Beginning center/cardinal augmentation after 39/64 of training will exceed 9,163 correct predictions by extending the monotonically beneficial inference-aligned phase beyond three-eighths while making a smaller change than the timed-out two-fifths configuration.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 74.47288516582921, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.24135459518432617, "validation_score": 9155.402785797016}
prior_hypothesis: Extending center/cardinal one-pixel augmentation from the final quarter to the final third of the validated batch-64 training run will exceed 9,141 correct predictions by increasing the EMA model’s exposure to inference-matched views while retaining broad ±2 translation regularization early in training.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 67.55882224999368, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.24300272483825683, "validation_score": 9141.402251732847}
prior_hypothesis: Using the validated batch-64 configuration while restricting the final quarter of training to the five center/cardinal one-pixel views used by inference will exceed 9,128 correct predictions by preserving early translation regularization and eliminating late optimization on the diagonal and two-pixel views that are mismatched with the best ensemble.

## Recent verification evidence

RECENT RESULT
hypothesis: Beginning center/cardinal one-pixel augmentation after 60% of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the unverified half-run configuration.
change: Run broad ±2 translation augmentation for the first three-fifths of training and center/cardinal augmentation for the final two-fifths.
mechanism: Intermediate-duration inference-aligned augmentation
evidence_used: Extending the cardinal phase from the final quarter to the final third improved validation correct from 9,141 to 9,155 and lowered cross-entropy; the attempted final-half phase timed out, making a two-fifths phase the closest informative intermediate test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Beginning center/cardinal one-pixel augmentation after five-eighths of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the timed-out two-fifths configuration.
change: Preserve the validated batch-64 model, optimizer, EMA, loss, and ten-view ensemble while using broad ±2 translations for the first five-eighths of training and center/cardinal one-pixel translations for the final three-eighths.
mechanism: Three-eighths inference-aligned augmentation phase
evidence_used: Extending the cardinal phase from the final quarter to the final third increased validation correct from 9,141 to 9,155 and reduced cross-entropy from 0.24300 to 0.24135; three-eighths is the untested midpoint between that best completed duration and the timed-out two-fifths attempt.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 68.43390987510793, "validation_accuracy": 0.9163, "validation_correct": 9163, "validation_cross_entropy": 0.24108177490234375, "validation_score": 9163.402874339235}

RECENT RESULT
hypothesis: Beginning center/cardinal augmentation after 39/64 of training will exceed 9,163 correct predictions by extending the monotonically beneficial inference-aligned phase beyond three-eighths while making a smaller change than the timed-out two-fifths configuration.
change: Adopt the best validated batch-64 architecture, optimizer, EMA, loss, and ten-view ensemble, then use broad ±2 translations for the first 39/64 of training and center/cardinal translations thereafter.
mechanism: Slightly extended inference-aligned augmentation phase
evidence_used: Extending the cardinal phase from one-quarter to one-third to three-eighths increased validation correct from 9,141 to 9,155 to 9,163; 25/64 is the smallest informative extension toward the unverified two-fifths phase.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245818, "training_seconds": 70.46581783308648, "validation_accuracy": 0.9167, "validation_correct": 9167, "validation_cross_entropy": 0.23885296478271484, "validation_score": 9167.403599147125}

RECENT RESULT
hypothesis: Beginning center/cardinal augmentation after 19/32 of training will exceed 9,167 correct predictions by continuing the observed monotonic benefit of longer inference-aligned phases while changing the validated boundary by only 1/64 of training.
change: Use broad ±2 translations for the first 19/32 of training and center/cardinal one-pixel translations for the final 13/32, retaining all other validated settings.
mechanism: Incrementally extended inference-aligned augmentation
evidence_used: Extending the cardinal phase from 1/4 to 1/3 to 3/8 to 25/64 increased validation correct from 9,141 to 9,155 to 9,163 to 9,167 and most recently reduced cross-entropy to 0.23885; 13/32 is the next conservative boundary extension.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Beginning cardinal augmentation after 77/128 of training will exceed 9,167 correct predictions by extending the beneficial inference-aligned phase beyond the best completed 25/64 phase without jumping fully to the timed-out 13/32 phase.
change: Use broad ±2 translations for the first 77/128 of training and center/cardinal one-pixel translations thereafter.
mechanism: Conservative extension of inference-aligned augmentation
evidence_used: Extending the cardinal phase from 1/4 through 25/64 monotonically improved correct predictions from 9,141 to 9,167; 77/128 is the midpoint between the best completed 25/64 duration and the timed-out 13/32 attempt.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Beginning cardinal augmentation after 155/256 of training will exceed 9,167 correct predictions by extending the consistently beneficial inference-aligned phase slightly beyond the best completed 25/64 configuration.
change: Use broad ±2 translations for the first 155/256 of training and center/cardinal one-pixel translations thereafter, retaining every other validated setting.
mechanism: Fine-grained extension of inference-aligned augmentation
evidence_used: Extending the cardinal phase from 1/4 through 25/64 monotonically improved validation correct from 9,141 to 9,167; 155/256 is the conservative midpoint between the best completed 39/64 boundary and the timed-out 77/128 boundary.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding identity-initialized, image-conditioned channel gates at each convolutional stage will exceed 9,167 correct predictions by injecting global shape context while preserving the validated spatially indexed classifier and convolution widths.
change: Use the best validated 39/64 augmentation boundary and add lightweight squeeze-and-excitation-style channel recalibration before each pooling operation, increasing the model to 249,618 learned parameters.
mechanism: Multi-scale global channel recalibration
evidence_used: Spatial attention pooling fell to 9,103 correct, indicating that replacing the position-sensitive head discards useful structure; this alternative retains that head and all validated compute-heavy layers while testing global contextual feature modulation with negligible additional spatial computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Giving the centered views twice the weight of each shifted view will exceed 9,167 correct predictions by retaining the demonstrated benefit of cardinal-view diversity while reducing dependence on displaced predictions.
change: Preserve the best validated training configuration and change the ten-view probability ensemble so each centered orientation has weight two, each shifted orientation has weight one, and the weighted mean uses total weight twelve.
mechanism: Center-weighted cardinal test-time augmentation
evidence_used: Cardinal ensembling improved flip-only evaluation from 9,110 to 9,125 correct, establishing that shifted views add useful evidence; center weighting directly tests whether their benefit comes from complementary predictions rather than requiring the current 80% aggregate weight on shifted images.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the best 39/64 augmentation boundary and weighting centered views 1.5× will exceed 9,167 correct predictions by preserving useful cardinal diversity while reducing shifted views’ aggregate influence from 80% to 72.7%.
change: Restore the validated 39/64 broad-to-cardinal training transition and assign each centered orientation weight 1.5 while retaining unit weight for all eight shifted views.
mechanism: Moderate center-weighted cardinal test-time augmentation
evidence_used: The 39/64 boundary achieved the best completed result at 9,167 correct, and cardinal ensembling previously improved flip-only evaluation from 9,110 to 9,125; the timed-out 2× center-weight experiment motivates a smaller intermediate reweighting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Doubling the EMA update rate during the inference-aligned cardinal phase will exceed 9,167 correct predictions by reducing stale broad-translation influence on final weights and BatchNorm statistics.
change: Restore the best 39/64 augmentation boundary and change EMA interpolation from 0.005 to 0.01 only after that boundary.
mechanism: Phase-aware EMA acceleration
evidence_used: Extending the cardinal phase from one-quarter through 25/64 monotonically raised validation correct from 9,141 to 9,167; faster late EMA tracking tests whether emphasizing that proven terminal distribution yields another gain.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the validated 39/64 augmentation transition and widening the flattened spatial head from 48 to 52 units will exceed 9,167 correct predictions by preserving more position-dependent feature combinations while remaining below the parameter ceiling.
change: Restore the best validated augmentation boundary and use the remaining parameter budget to expand the classifier to 249,318 learned parameters.
mechanism: Expanded position-sensitive classifier bottleneck
evidence_used: The 39/64 transition achieved the best completed result at 9,167 correct, while spatial attention pooling fell to 9,103, indicating that the position-sensitive flattened head is valuable and motivating a capacity increase that preserves it.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a residual path around the second convolution of each stage will exceed 9,167 correct predictions by improving feature reuse and optimization during the fixed 100,000-example exposure.
change: Preserve the validated 39/64 augmentation curriculum, spatial classifier, optimizer, EMA, loss, and test-time ensemble while converting each convolution pair into a residual stage with unchanged learned-parameter count.
mechanism: Parameter-neutral residual feature refinement
evidence_used: Spatial attention pooling reduced validation correct to 9,103, indicating that the position-sensitive head should be retained; this patch instead improves the feature extractor’s optimization topology without consuming the remaining parameter budget or adding substantial computation.
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
