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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.87196491705254, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.20446708946228026, "validation_score": 9315.415121346507}
prior_hypothesis: Restoring translation-free inputs and shortening warmup from 5% to 2% will exceed 9,330 correct predictions by providing more peak-rate optimization within the fixed exposure while preserving the winning late-training schedule.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 76.88440129184164, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.2005714702606201, "validation_score": 9330.416468333944}
prior_hypothesis: Scaling ensemble logits by 1.05 during evaluation will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.204606.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.32190362503752, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Scaling evaluation logits by 1.15 will preserve the current 9,330 correct predictions while reducing validation cross-entropy from 0.198018 to approximately 0.196639.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 80.97804520791396, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.19663925247192382, "validation_score": 9330.41783687019}
prior_hypothesis: Restoring the linear ensemble curriculum and increasing evaluation scaling to 1.15 will retain all 9,330 argmax predictions while reducing validation cross-entropy below 0.198018.

## Recent verification evidence

RECENT RESULT
hypothesis: Restoring the proven 5% warmup and 1.15 evaluation scale while annealing label smoothing from 0.02 to zero will exceed 9,330 correct predictions by retaining early regularization and sharpening decision boundaries late in the fixed exposure.
change: Restore the best verified schedule and calibration, then linearly decay label smoothing throughout training without adding parameters or computation.
mechanism: Late hard-target annealing for symmetric flip-ensemble learning
evidence_used: The 5% warmup with symmetric logit fusion reached 9,330 correct, while 2% warmup reached 9,315; alternative decision-time fusion fell to 9,319 or 9,326, so this tests a training-time margin change while preserving the best verified fusion rule.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 71.6606682499405, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.19499085655212403, "validation_score": 9312.418413243297}

RECENT RESULT
hypothesis: Adding a small late-ramped Jensen–Shannon penalty between original and flipped predictions will exceed 9,330 correct predictions by reducing view disagreement while preserving the best verified equal-logit ensemble and fixed label smoothing.
change: Retain the proven training and evaluation procedure, adding parameter-free flip-view consistency that increases from zero to 0.05 over training.
mechanism: Ramped symmetric flip-view consistency
evidence_used: Alternative evaluation fusion reduced accuracy to 9,319 or 9,326, and annealed label smoothing reduced it to 9,312; this motivates preserving the verified decision rule and fixed targets while improving agreement between its two learned views.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging model parameters over the final half of training with decay 0.99 will exceed 9,330 correct predictions by reducing optimizer noise while preserving the proven training loss, schedule, and equal-logit ensemble.
change: Maintain a parameter EMA after the midpoint of training, use it during evaluation, and restore online parameters if training resumes.
mechanism: Late-trajectory exponential weight averaging
evidence_used: Decision-time probability fusion reduced correctness to 9,326, canonical weighting reduced it to 9,319, and label-smoothing annealing reduced it to 9,312; this motivates retaining the best verified decision and target procedures while improving the final learned parameter estimate.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.43282191711478, "validation_accuracy": 0.9327, "validation_correct": 9327, "validation_cross_entropy": 0.19723106460571288, "validation_score": 9327.41763032616}

RECENT RESULT
hypothesis: Capping the ensemble-loss curriculum at 0.90 will exceed 9,330 correct predictions by preserving independent supervision for both views late in training; if correctness ties, the verified 1.15 evaluation scale will reduce cross-entropy.
change: Preserve the proven architecture, optimizer, fixed label smoothing, and symmetric logit fusion, while retaining 10% per-view loss at the curriculum endpoint and restoring the best verified evaluation scale.
mechanism: Residual per-view supervision with verified logit calibration
evidence_used: Equal-logit fusion achieved 9,330 correct, while probability fusion, orientation-biased fusion, EMA, and annealed label smoothing achieved 9,326, 9,319, 9,327, and 9,312 respectively; this motivates keeping the winning inference and target procedures while modestly strengthening view-specific training. Scale 1.15 preserved all 9,330 predictions and improved cross-entropy to 0.196639.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Ending the ensemble-loss curriculum at 0.90 will exceed 9,330 correct predictions by preserving useful per-view supervision, while the verified 5% warmup and 1.15 logit scale retain the strongest established schedule and calibration.
change: Restore the best verified warmup and evaluation scale, and retain 10% per-view loss at the curriculum endpoint.
mechanism: Residual per-view supervision with calibrated symmetric ensembling
evidence_used: The verified 5% warmup and 1.15-scaled equal-logit ensemble achieved 9,330 correct with 0.196639 cross-entropy; alternative fusion, EMA, and smoothing changes performed worse. The previous residual-supervision verification timed out, providing no negative model evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising fixed label smoothing from 0.02 to 0.03 will exceed 9,330 correct predictions by strengthening the regularization that was lost when smoothing was annealed toward zero.
change: Preserve the verified architecture, optimizer, curriculum, schedule, and evaluation ensemble while increasing label smoothing for both ensemble and per-view losses.
mechanism: Slightly stronger fixed label smoothing
evidence_used: Annealing smoothing from 0.02 to zero reduced validation correctness from 9,330 to 9,312 despite lower cross-entropy, indicating that maintaining soft targets benefits the primary accuracy objective.
result: training did not finish within the verification time limit

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
