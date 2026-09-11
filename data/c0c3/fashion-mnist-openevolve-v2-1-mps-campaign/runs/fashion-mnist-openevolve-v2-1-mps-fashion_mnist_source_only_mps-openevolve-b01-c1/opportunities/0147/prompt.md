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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.86444654199295, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2156680419921875, "validation_score": 9267.41129649109}
prior_hypothesis: Increasing the geometric component from 2% to 3% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215711, thereby exceeding validation_score 9267.411282.

## Recent verification evidence

RECENT RESULT
hypothesis: Lowering the cosine schedule’s final learning rate from 9e-5 to 3e-5 will exceed 9,262 correct predictions by reducing late-update noise while preserving the successful early trajectory.
change: Change only the cosine learning-rate floor from 3% to 1% of the 3e-3 peak.
mechanism: Deeper terminal learning-rate annealing
evidence_used: The best architecture and 0.020→0.009 smoothing schedule remain strongest, while capacity changes regressed and late EMA fell to 9,244 correct; a more convergent endpoint directly tests late optimization stability without extra parameters, steps, or runtime-heavy averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 52.30452495883219, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.21324563941955565, "validation_score": 9253.412117697979}

RECENT RESULT
hypothesis: Raising the cosine learning-rate floor from 3% to 5% of the 3e-3 peak will exceed 9,262 correct predictions by preserving useful late-training plasticity during the fixed two-pass exposure budget.
change: Change only the cosine schedule’s terminal learning-rate floor from 9e-5 to 1.5e-4.
mechanism: Higher terminal learning-rate floor
evidence_used: Lowering the floor from 3% to 1% regressed from 9,262 to 9,253 correct and worsened cross-entropy from 0.21199 to 0.21325, motivating the informative opposite-direction test while preserving the strongest architecture, optimizer, and smoothing schedule.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 44.822912541916594, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21194436264038086, "validation_score": 9265.412560192872}

RECENT RESULT
hypothesis: Raising the cosine learning-rate floor from 5% to 7% of the 3e-3 peak will exceed 9,265 correct predictions by sustaining beneficial late-training plasticity during the fixed exposure budget.
change: Change only the cosine schedule’s terminal learning-rate floor from 1.5e-4 to 2.1e-4.
mechanism: Extended terminal learning-rate floor
evidence_used: Increasing the floor from 1% to 3% improved correct predictions from 9,253 to 9,262, and increasing it again to 5% improved them to 9,265 with slightly better cross-entropy; an equal-sized step to 7% is the clearest test of the observed trend.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.328348458046094, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.21339555015563966, "validation_score": 9252.412066782292}

RECENT RESULT
hypothesis: Lowering the cosine learning-rate floor from 5% to 4% of the 3e-3 peak will exceed 9,265 correct predictions by retaining more late plasticity than the 3% floor while avoiding the instability observed at 7%.
change: Change only the terminal learning-rate floor from 1.5e-4 to 1.2e-4.
mechanism: Refined terminal learning-rate annealing
evidence_used: The 5% floor achieved 9,265 correct versus 9,262 at 3%, while 7% regressed sharply to 9,252; testing 4% is the most promising local refinement on the less aggressive side of the current optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 50.46227295813151, "validation_accuracy": 0.9261, "validation_correct": 9261, "validation_cross_entropy": 0.21175265731811524, "validation_score": 9261.412625461953}

RECENT RESULT
hypothesis: A 4.5% cosine learning-rate floor will exceed 9265.412560 by retaining the 5% floor’s classification accuracy while moving cross-entropy toward the better 0.211753 achieved at 4%.
change: Lower the terminal learning-rate multiplier from 0.05 to 0.045 while preserving the 3e-3 peak and all other behavior.
mechanism: Half-step terminal learning-rate refinement
evidence_used: The 5% floor produced the best result with 9,265 correct; 4% was only four predictions behind and had lower cross-entropy, whereas 7% regressed by thirteen predictions. The midpoint between 4% and 5% is therefore the strongest remaining local refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 60.95691333292052, "validation_accuracy": 0.9262, "validation_correct": 9262, "validation_cross_entropy": 0.21171827087402345, "validation_score": 9262.412637171543}

RECENT RESULT
hypothesis: Adding one lightweight self-attention block over the 7×7 feature grid will exceed 9,265 correct predictions by learning image-conditioned long-range relationships between garment parts while retaining coordinate-specific classification.
change: Insert learned positional embeddings and a pre-normalized four-head attention/MLP block before the flattened classifier, narrowing its bottleneck from 64 to 59 to keep the model at approximately 248,097 parameters.
mechanism: Position-aware global token attention
evidence_used: Position-free pooling alternatives regressed to 9,253 and 9,249, showing spatial layout matters, while widening the static flattened head regressed to 9,221. This challenges the load-bearing assumption that local convolutions followed by fixed flattened mixing are sufficient, using content-dependent global mixing without the expensive extra convolutional block that timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Arithmetic probability pooling across shifted and flipped views will exceed 9,265 correct predictions by preventing a single overconfident crop from dominating the ensemble.
change: Softmax-normalize each augmentation prediction before averaging, then return scaled log-probabilities as ten-class logits.
mechanism: Confidence-normalized test-time augmentation pooling
evidence_used: The 233,434-parameter architecture with the 5% learning-rate floor remains best, while capacity changes, EMA, and attention failed or regressed; this preserves that training trajectory and tests the previously unchanged aggregation of its existing 50 validation views.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 67.3351551250089, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21580071029663087, "validation_score": 9266.41125161037}

RECENT RESULT
hypothesis: Training the dominant ensemble objective with arithmetic probability pooling will exceed 9,266 correct predictions by aligning optimization with the validation aggregation that produced the current best result.
change: Replace the geometric logit-average ensemble loss with a weighted arithmetic mean of per-view probabilities, converted back to log-probability logits for cross-entropy.
mechanism: Train-time probability-ensemble alignment
evidence_used: Switching validation aggregation to arithmetic probability pooling improved the best result from 9,265 to 9,266 correct, but training still optimizes a logit-pooled ensemble; removing this train/evaluation mismatch directly tests the mechanism responsible for the improvement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 62.516105625079945, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.2167792423248291, "validation_score": 9229.410920882448}

RECENT RESULT
hypothesis: Blending 10% equal hard votes into the successful arithmetic probability ensemble will exceed 9,266 correct predictions by reducing residual sensitivity to differing confidence magnitudes across transformed views.
change: Keep training unchanged and add a small one-hot vote component to each validation-view probability vector before the existing spatial aggregation.
mechanism: Soft-probability and plurality-vote TTA hybrid
evidence_used: Arithmetic probability pooling improved validation correct from 9,265 to 9,266, while aligning the training loss to probability pooling regressed to 9,229; this motivates refining only inference aggregation toward confidence-independent voting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.20351720904, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.21567442932128905, "validation_score": 9264.411294330077}

RECENT RESULT
hypothesis: Blending 1% of the prior geometric logit ensemble into arithmetic probability pooling will retain 9,266 correct predictions while lowering cross-entropy below 0.215801, exceeding validation_score 9266.411252.
change: Accumulate validation-view logits alongside probabilities and interpolate 99% of the calibrated arithmetic log-probabilities with 1% of the geometric ensemble’s normalized logits.
mechanism: Near-arithmetic log-opinion pooling
evidence_used: Arithmetic pooling improved correct predictions from 9,265 to 9,266 but worsened cross-entropy from 0.211944 to 0.215801; a small interpolation toward the better-calibrated geometric endpoint is likely to improve the tie-breaker without crossing enough decision boundaries to lose the accuracy gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 62.29763370892033, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.2157552589416504, "validation_score": 9266.41126698513}

RECENT RESULT
hypothesis: Increasing the geometric component from 1% to 2% will retain 9,266 correct predictions while lowering cross-entropy below 0.215755, exceeding validation_score 9266.411267.
change: Interpolate 98% calibrated arithmetic log-probabilities with 2% normalized geometric logits during validation inference.
mechanism: Two-percent geometric log-opinion blend
evidence_used: Adding a 1% geometric component preserved the arithmetic ensemble’s 9,266 correct predictions and improved cross-entropy from 0.215801 to 0.215755; the geometric endpoint’s substantially lower 0.211944 cross-entropy motivates one more conservative step in the same direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 70.40756420791149, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21571103439331055, "validation_score": 9267.411281946}

RECENT RESULT
hypothesis: Increasing the geometric component from 2% to 3% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215711, thereby exceeding validation_score 9267.411282.
change: Interpolate 97% calibrated arithmetic log-probabilities with 3% normalized geometric logits during validation inference.
mechanism: Three-percent geometric log-opinion blend
evidence_used: Moving from 1% to 2% geometric blending improved validation correct from 9,266 to 9,267 and reduced cross-entropy from 0.215755 to 0.215711; the geometric endpoint also has substantially lower cross-entropy, motivating one further conservative step.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 49.86444654199295, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2156680419921875, "validation_score": 9267.41129649109}



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
