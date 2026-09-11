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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 76.39047762518749, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21636803817749023, "validation_score": 9249.411059797945}
prior_hypothesis: Averaging per-view class probabilities will exceed 9,240 correct predictions, or tie while lowering cross-entropy, by preventing an overconfident translated or flipped view from dominating the ensemble’s decision.

## Recent verification evidence

RECENT RESULT
hypothesis: Setting the peak learning rate to 3.15e-3 will exceed 9,240 correct predictions by retaining most of the optimization benefit of 3.3e-3 while reducing its apparent late-stage overshoot.
change: Reduce only the peak learning rate from 3.3e-3 to the midpoint between the two verified rates, preserving the successful batch size, warmup, architecture, augmentation, and inference ensemble.
mechanism: Interpolated peak-rate calibration
evidence_used: At batch size 96, increasing the peak rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened cross-entropy from 0.21392 to 0.21460; the midpoint directly tests whether the accuracy gain can be retained with better convergence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 66.90865995781496, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.21553845291137697, "validation_score": 9227.411340339586}

RECENT RESULT
hypothesis: Replacing the fixed-position 30-feature bottleneck with seven learned spatial part summaries plus a global summary will exceed 9,240 correct predictions by preserving substantially more channel information while making prediction less dependent on exact 7×7 coordinates.
change: Replace the 94,420-parameter flattened classifier with a 235,793-total-parameter attention-pooling model that dynamically locates seven discriminative regions, retains a global feature token, and classifies their concatenated 512-feature representation.
mechanism: Content-addressed multi-part attention pooling
evidence_used: The strongest gains came from translation-matched training and translated-view inference, indicating that limited position invariance is valuable, while the current head remains load-bearing on exact spatial coordinates and compresses 3,136 activations to only 30. This tests the alternative assumption that content-addressed garment parts are more useful than a fixed-location bottleneck without the convolutional overhead that caused attention-gated and widened designs to time out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 235793, "training_seconds": 67.15439787507057, "validation_accuracy": 0.9228, "validation_correct": 9228, "validation_cross_entropy": 0.21531134490966797, "validation_score": 9228.411417207693}

RECENT RESULT
hypothesis: Evaluating a 0.99-decay EMA of the trained weights will exceed 9,240 correct predictions by preserving the successful 3.3e-3 optimization trajectory while reducing endpoint noise.
change: Maintain an exponential moving average of model parameters and floating-point BatchNorm buffers after every optimizer step, and use those averaged values for validation inference without adding learned parameters.
mechanism: Late-trajectory exponential weight averaging
evidence_used: Raising the peak rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened cross-entropy from 0.21392 to 0.21460, suggesting useful optimization progress with a noisier final iterate; EMA directly stabilizes that iterate while retaining the verified schedule and batch size.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying 0.02 label smoothing will exceed 9,240 correct predictions by retaining the successful 3.3e-3 optimization trajectory while reducing brittle, overconfident decision boundaries.
change: Add 0.02 label smoothing to the training cross-entropy without changing architecture, exposure, schedule, or runtime materially.
mechanism: Mild confidence-regularized cross-entropy
evidence_used: Raising the peak learning rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened validation cross-entropy from 0.21392 to 0.21460, motivating a small confidence regularizer that preserves the stronger optimizer setting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing the classifier dropout will exceed 9,240 correct predictions by allowing faster fitting during the fixed two-pass exposure budget.
change: Replace the 0.1 dropout layer with an identity operation, preserving architecture size, runtime, augmentation, and optimization.
mechanism: Low-exposure classifier de-regularization
evidence_used: Increasing optimizer updates with batch size 96 improved correctness from 9,204 to 9,239, and raising the peak learning rate gained another prediction, indicating optimization-limited training; removing dropout should similarly increase effective learning without the timeout risk of additional steps.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging per-view class probabilities will exceed 9,240 correct predictions, or tie while lowering cross-entropy, by preventing an overconfident translated or flipped view from dominating the ensemble’s decision.
change: Preserve the verified model and training procedure, but replace evaluation-time logit averaging with weighted probability averaging and return the resulting log-probabilities.
mechanism: Probability-space test-time augmentation fusion
evidence_used: The strongest design relies on center/cardinal translated-view inference, while the successful 3.3e-3 learning rate slightly worsened cross-entropy; bounded probability fusion directly targets ensemble robustness and calibration without changing training time, learned parameters, or optimizer behavior.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 76.39047762518749, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21636803817749023, "validation_score": 9249.411059797945}

RECENT RESULT
hypothesis: Scaling the fused log-probabilities by 1.1 will retain exactly 9,249 correct predictions while lowering validation cross-entropy below 0.216368.
change: Apply mild temperature sharpening after the verified probability-space test-time augmentation, without changing its class decisions, training, or runtime materially.
mechanism: Temperature-sharpened probability fusion
evidence_used: Probability averaging improved correctness from 9,240 to 9,249 but worsened cross-entropy from 0.214599 to 0.216368; positive logit scaling preserves every argmax while testing whether the arithmetic ensemble is slightly underconfident.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 77.51782941585407, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.21500194969177247, "validation_score": 9249.411521973381}

RECENT RESULT
hypothesis: Softening each view’s logits by 0.9 before probability averaging will exceed 9,249 correct predictions by further limiting confident-view dominance; if correctness ties, scaling the fused log-probabilities by 1.1 will lower cross-entropy.
change: Apply temperature softening inside every test-time view softmax, then apply the previously favorable 1.1 sharpening only after probability fusion.
mechanism: Decoupled per-view softening and ensemble sharpening
evidence_used: Probability-space fusion improved correctness from 9,240 to 9,249, supporting less confidence-dominated view aggregation; subsequent 1.1 fused-output sharpening retained all 9,249 correct predictions while reducing cross-entropy from 0.216368 to 0.215002.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling each view’s logits by 0.95 before probability averaging will exceed 9,249 correct predictions by further reducing outlier-view dominance; if correctness ties, 1.1 post-fusion sharpening will lower cross-entropy below 0.216368.
change: Mildly soften every test-time view distribution before fusion, then sharpen only the fused log-probabilities without changing their argmax.
mechanism: Conservative per-view temperature softening with calibrated fusion
evidence_used: Probability averaging improved correctness from 9,240 to 9,249, while 1.1 post-fusion sharpening retained all 9,249 predictions and reduced cross-entropy to 0.215002; 0.95 is a conservative intermediate test after the unmeasured 0.9-softening verification timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the fused log-probabilities by 1.1 will preserve all 9,249 class decisions while lowering validation cross-entropy below 0.216368.
change: Sharpen the probability-space test-time ensemble after fusion without changing training or predicted argmaxes.
mechanism: Post-fusion temperature sharpening
evidence_used: The prior 1.1-scaling attempt reported 9,249 correct with cross-entropy reduced to 0.215002; although that run exceeded the time limit, its measurements directly support retrying this computation-neutral calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the fixed-position bottleneck from 30 to 31 features will exceed 9,249 correct predictions by adding one spatial template while retaining the location-sensitive head that outperformed attention pooling.
change: Use the remaining parameter budget to widen the classifier bottleneck by one feature, increasing parameters from 245,044 to 248,191 without materially changing runtime.
mechanism: Budget-maximized fixed spatial bottleneck
evidence_used: Replacing the 30-feature fixed-position head with content-addressed attention pooling reduced correctness to 9,228, indicating that fixed spatial summaries are valuable; a controlled one-feature widening tests additional capacity within that successful representation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 248191, "training_seconds": 61.635926167014986, "validation_accuracy": 0.9213, "validation_correct": 9213, "validation_cross_entropy": 0.2168298568725586, "validation_score": 9213.410903790022}

RECENT RESULT
hypothesis: Injecting vertical position and horizontal distance-from-center into the convolutional trunk will exceed 9,249 correct predictions by letting early filters distinguish geometrically different garment regions.
change: Replace the assumption that shared local filters should remain position-agnostic until the flattened head with fixed, flip-symmetric coordinate channels mixed into every first-layer feature.
mechanism: Flip-symmetric coordinate-conditioned convolution
evidence_used: Content-addressed attention pooling fell to 9,228 correct, while widening the fixed-position bottleneck fell to 9,213; this suggests spatial geometry is load-bearing but additional late-head capacity is not, motivating earlier position-conditioned feature extraction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245620, "training_seconds": 73.67292825016193, "validation_accuracy": 0.9212, "validation_correct": 9212, "validation_cross_entropy": 0.21608441314697266, "validation_score": 9212.411155668631}



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
