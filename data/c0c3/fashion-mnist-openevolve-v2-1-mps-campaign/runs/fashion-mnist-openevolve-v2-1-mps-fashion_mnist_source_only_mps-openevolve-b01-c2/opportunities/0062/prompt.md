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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.32190362503752, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Scaling evaluation logits by 1.15 will preserve the current 9,330 correct predictions while reducing validation cross-entropy from 0.198018 to approximately 0.196639.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.72896141698584, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Restoring the verified 5% warmup and 1.15 evaluation scale will increase validation correctness from 9,315 to 9,330 and reduce cross-entropy toward 0.196639.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.88440129184164, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.2005714702606201, "validation_score": 9330.416468333944}
prior_hypothesis: Scaling ensemble logits by 1.05 during evaluation will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.204606.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.97804520791396, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Restoring the linear ensemble curriculum and increasing evaluation scaling to 1.15 will retain all 9,330 argmax predictions while reducing validation cross-entropy below 0.198018.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.18 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.15 to 1.18.
mechanism: Fine-grained evaluation temperature sharpening
evidence_used: Cross-entropy improved from 0.200571 at scale 1.05 to 0.198018 at 1.10 and 0.196639 at 1.15 without changing correctness; the diminishing gains imply a calibration optimum near 1.18.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing fixed label smoothing from 0.02 to 0.03 will exceed 9,330 correct predictions, while the verified 1.15 evaluation scale will minimize cross-entropy if correctness ties.
change: Raise label smoothing for both ensemble and per-view losses to 0.03 and restore the best verified symmetric-ensemble scale of 1.15.
mechanism: Stronger fixed target smoothing with proven ensemble calibration
evidence_used: Annealing smoothing toward zero reduced correctness from 9,330 to 9,312, indicating soft targets support accuracy; the prior 0.03 verification timed out without negative model evidence, while scale 1.15 preserved 9,330 predictions and improved cross-entropy to 0.196639.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding lightweight channel attention while restoring the verified 5% warmup and 1.15 evaluation scale will exceed 9,330 correct predictions by adaptively emphasizing useful residual features without increasing computation enough to risk the time limit.
change: Add a squeeze-and-excitation gate to the 7×7 residual branch, reduce the classifier width from 35 to 34 to remain below 250,000 parameters, and restore the best verified schedule and calibration.
mechanism: Capacity-neutral residual channel recalibration
evidence_used: The 5% warmup and 1.15-scaled symmetric ensemble achieved 9,330 correct, while changes to fusion, EMA, and target scheduling were worse; this motivates preserving the proven training procedure and testing a parameter-efficient architectural improvement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.18 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.15 to 1.18.
mechanism: Fine-grained evaluation temperature sharpening
evidence_used: Scaling from 1.05 to 1.10 to 1.15 preserved 9,330 correct predictions while cross-entropy improved from 0.200571 to 0.198018 to 0.196639; the prior 1.18 verification timed out and therefore provides no negative validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reallocating residual-convolution parameters to widen the classifier bottleneck from 35 to 44 will exceed 9,330 correct predictions while remaining under 250,000 parameters.
change: Factor both 64-channel residual convolutions into two-group spatial convolutions followed by pointwise mixing, and spend the saved capacity on the classifier head; all proven training and inference procedures remain unchanged.
mechanism: Group-factorized residual refinement with a wider spatial classifier
evidence_used: The current architecture and symmetric logit ensemble reached 9,330 correct, while changes to fusion, smoothing, and EMA were worse; this motivates preserving that procedure and testing a capacity reallocation within the architecture.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.15 will preserve all 9,330 argmax predictions while reducing validation cross-entropy from 0.200571 to approximately the verified 0.196639.
change: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.05 to 1.15.
mechanism: Verified symmetric-ensemble temperature sharpening
evidence_used: Reference Design 2 used the identical training procedure with scale 1.15 and achieved 9,330 correct with 0.196639 cross-entropy, strictly outperforming the current 9,330 correct with 0.200571 cross-entropy.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the verified 5% warmup and 1.15 evaluation scale will increase validation correctness from 9,315 to 9,330 and reduce cross-entropy toward 0.196639.
change: Replace the regressed 2% warmup with 5% and apply the proven 1.15 temperature scale to evaluation-time averaged logits.
mechanism: Verified warmup restoration with calibrated symmetric flip ensembling
evidence_used: Reference Design 2 achieved 9,330 correct with 0.196639 cross-entropy using these exact settings, versus the current design’s 9,315 correct and 0.204467 cross-entropy.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.72896141698584, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}

RECENT RESULT
hypothesis: Scaling ensemble logits by 1.184 will preserve all 9,330 argmax predictions and reduce validation cross-entropy below 0.196639.
change: Change only the evaluation-time symmetric ensemble scale from 1.15 to 1.184.
mechanism: Quadratic-fit evaluation temperature calibration
evidence_used: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while cross-entropy fell from 0.200571 to 0.198018 to 0.196639; their diminishing improvements place the estimated calibration optimum near 1.184.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training on centered images plus four one-pixel translations will exceed 9,330 correct predictions by improving local shift invariance while preserving the verified architecture, optimization, flip ensemble, and calibration.
change: Cycle batches through centered, up, down, left, and right one-pixel crops using zero padding, without increasing parameter count or model-forward cost.
mechanism: Deterministic one-pixel translation augmentation
evidence_used: Symmetric flip training and inference achieved the best verified result of 9,330 correct, while altered fusion, EMA, and smoothing reduced correctness; this motivates preserving the proven procedure and adding a lightweight spatial-invariance augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the batch size to 80 while scaling all learning rates by 80/64 will complete within the verification limit and match or exceed 9,330 correct predictions.
change: Use batch size 80 and proportionally scale the AdamW initialization and cosine schedule peak learning rates to preserve learning-rate exposure per processed example.
mechanism: Sample-normalized larger-batch optimization
evidence_used: The verified design required 80.73 seconds, while subsequent even scale-only variants repeatedly timed out; reducing optimizer steps from roughly 1,564 to 1,250 directly targets this constraint while retaining the proven architecture, loss, augmentation, schedule shape, and evaluation calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.05 to 1.15 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196639.
change: Change only the evaluation-time symmetric flip-ensemble logit scale to the best verified value.
mechanism: Verified symmetric-ensemble temperature sharpening
evidence_used: Reference Design 1 differs from the current design only in this scale and achieved the same 9,330 correct predictions with lower cross-entropy, improving validation_score from 9330.416468 to 9330.417837; later timeout results provide no contrary validation evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Allowing each residual channel’s contribution to adapt from the verified identity-equivalent initialization will exceed 9,330 correct predictions without materially increasing runtime.
change: Add 64 learned channel scales initialized to one and apply them to the residual branch before addition, preserving the current model’s initial function and all verified training and evaluation settings.
mechanism: Identity-initialized per-channel residual scaling
evidence_used: The current architecture and procedure repeatedly achieved 9,330 correct, while the heavier channel-attention experiment timed out; identity-initialized channel scaling provides similar adaptive recalibration with negligible computation and only 64 additional parameters.
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
