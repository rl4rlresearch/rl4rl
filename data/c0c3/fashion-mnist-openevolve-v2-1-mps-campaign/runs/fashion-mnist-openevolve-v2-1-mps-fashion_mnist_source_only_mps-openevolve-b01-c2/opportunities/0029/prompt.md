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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 73.88488316698931, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.20460558624267577, "validation_score": 9330.415073618877}
prior_hypothesis: Annealing from equal ensemble/view supervision to pure ensemble supervision, while preserving the successful 75% average ensemble weight, will exceed 9,325 correct predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.87196491705254, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.20446708946228026, "validation_score": 9315.415121346507}
prior_hypothesis: Restoring translation-free inputs and shortening warmup from 5% to 2% will exceed 9,330 correct predictions by providing more peak-rate optimization within the fixed exposure while preserving the winning late-training schedule.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.3411985419225, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.20006971549987793, "validation_score": 9322.416642461301}
prior_hypothesis: Restoring the best batch-64 ensemble-loss curriculum and annealing label smoothing only during the final quarter will exceed 9,330 correct predictions by preserving early regularization while better matching hard-label validation late in training.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 78.42507937480696, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.2035751304626465, "validation_score": 9322.415428989305}
prior_hypothesis: A cosine shift from equal view/ensemble supervision to pure ensemble supervision will exceed 9,330 correct predictions by preserving view guidance longer early and emphasizing validation-matched ensembling more strongly late.

## Recent verification evidence

RECENT RESULT
hypothesis: Annealing from equal ensemble/view supervision to pure ensemble supervision, while preserving the successful 75% average ensemble weight, will exceed 9,325 correct predictions.
change: Linearly increase the ensemble-loss weight from 0.5 to 1.0 across training and decrease the per-view weight correspondingly.
mechanism: Validation-aligned flip-loss curriculum
evidence_used: Static 75% ensemble weighting reported 9,325 correct versus 9,312 for equal weighting, while earlier per-view supervision improved 9,261 to 9,276; this schedule retains per-view guidance early and emphasizes the validation-matched ensemble late.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 73.88488316698931, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.20460558624267577, "validation_score": 9330.415073618877}

RECENT RESULT
hypothesis: Restoring the best batch-64 ensemble-loss curriculum and annealing label smoothing only during the final quarter will exceed 9,330 correct predictions by preserving early regularization while better matching hard-label validation late in training.
change: Use batch size 64, linearly shift supervision from equal view/ensemble loss to pure ensemble loss, and decay label smoothing from 0.02 to zero over the final 25% of steps.
mechanism: Terminal hard-label ensemble fine-tuning
evidence_used: Linear ensemble-weight annealing achieved 9,330 correct versus 9,325 for static 75% weighting and 9,312 for equal weighting, showing that increasingly validation-aligned late supervision is beneficial; terminal smoothing decay extends that mechanism without removing early regularization.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.3411985419225, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.20006971549987793, "validation_score": 9322.416642461301}

RECENT RESULT
hypothesis: A cosine shift from equal view/ensemble supervision to pure ensemble supervision will exceed 9,330 correct predictions by preserving view guidance longer early and emphasizing validation-matched ensembling more strongly late.
change: Restore the best batch-64 ungated residual architecture and static 0.02 label smoothing, while replacing linear ensemble-weight annealing with a same-endpoint, same-average cosine curriculum.
mechanism: Cosine flip-loss curriculum
evidence_used: Linear ensemble weighting achieved the best 9,330 correct versus 9,325 for static 75% weighting, while terminal label-smoothing decay reduced accuracy to 9,322; this isolates the curriculum shape while retaining the proven regularization.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 78.42507937480696, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.2035751304626465, "validation_score": 9322.415428989305}

RECENT RESULT
hypothesis: Adding random ±2-pixel translations to the best linear flip-loss curriculum will exceed 9,330 correct predictions by improving local translation robustness without changing its proven architecture or inference rule.
change: Adopt Reference Design 1 and apply independently sampled two-dimensional translations during batch preparation while retaining its batch size, residual model, optimizer, linear ensemble-loss curriculum, and label smoothing.
mechanism: Per-example random translation with validation-aligned flip ensembling
evidence_used: Linear ensemble-loss annealing achieved the best result of 9,330 correct, while subsequent smoothing and curriculum-shape changes fell to 9,322; this motivates preserving the winning design and testing an orthogonal, previously untested input-invariance mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 79.45966495899484, "validation_accuracy": 0.9222, "validation_correct": 9222, "validation_cross_entropy": 0.22399459381103515, "validation_score": 9222.408498536292}

RECENT RESULT
hypothesis: Raising the cosine schedule’s minimum learning rate from 2% to 5% of peak will exceed 9,330 correct predictions by giving the increasingly validation-aligned ensemble objective more optimization leverage late in training.
change: Preserve the best architecture, loss curriculum, and peak learning rate while increasing only the cosine decay floor from 0.02 to 0.05.
mechanism: Curriculum-coupled late learning-rate floor
evidence_used: Linear ensemble-loss annealing achieved the best result of 9,330 correct, while alternative loss schedules reduced accuracy; the current learning-rate schedule decays to just 2% precisely when ensemble supervision becomes dominant, motivating an isolated increase in late optimization strength.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 69.91589570883662, "validation_accuracy": 0.9317, "validation_correct": 9317, "validation_cross_entropy": 0.2043524269104004, "validation_score": 9317.415160868884}

RECENT RESULT
hypothesis: Restoring static 0.02 label smoothing and lowering the cosine learning-rate floor from 2% to zero will exceed 9,330 correct predictions by letting the ensemble-dominant objective settle more fully late in training.
change: Restore Reference Design 2’s winning linear loss curriculum and static smoothing, then change only its cosine-decay floor from 0.02 to 0.
mechanism: Zero-floor cosine convergence
evidence_used: Linear ensemble annealing with static 0.02 smoothing achieved 9,330 correct; terminal smoothing decay fell to 9,322, and raising the learning-rate floor from 2% to 5% fell further to 9,317, motivating restoration of static smoothing and testing the opposite floor direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying a 0.995-decay parameter EMA to the best linear flip-loss curriculum will exceed 9,330 correct predictions by reducing late-training weight noise without altering exposure or inference cost.
change: Restore linear ensemble-loss annealing and replace the final learned parameters with their optimizer-step EMA before validation.
mechanism: Late-trajectory exponential weight averaging
evidence_used: Linear ensemble annealing achieved the best result at 9,330 correct, while cosine weighting fell to 9,322 and learning-rate-floor changes failed or regressed; this preserves the winning curriculum and tests an orthogonal stabilization mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the 9,330-correct linear curriculum and learning a class-specific convex weighting of original and flipped logits will exceed 9,330 correct by adapting view trust while preserving the proven logit-space ensemble.
change: Remove harmful translation augmentation and add ten sigmoid-constrained fusion parameters, initialized to reproduce equal averaging and trained through the existing ensemble loss.
mechanism: Learned classwise logit-space flip fusion
evidence_used: Reference Design 1 achieved 9,330 correct, whereas ±2 translations fell to 9,222 and predictive-probability fusion fell to 9,297; this restores the winner and retains arithmetic-logit aggregation while making its view weights learnable.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing label smoothing linearly from 0.01 to 0.03 while preserving its 0.02 average will exceed 9,330 correct predictions by concentrating regularization late, when supervision becomes purely ensemble-aligned.
change: Retain the winning architecture, optimizer, and linear ensemble-loss curriculum, but replace static 0.02 label smoothing with a same-average linear ramp.
mechanism: Back-loaded label-smoothing curriculum
evidence_used: Static 0.02 smoothing with linear ensemble annealing achieved 9,330 correct, whereas decaying smoothing during the final quarter fell to 9,322; this suggests late smoothing is valuable and motivates testing the opposite temporal allocation without changing average strength.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.45502533297986, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.21014063186645507, "validation_score": 9310.413175119349}

RECENT RESULT
hypothesis: Averaging parameter iterates over the final 10% of the winning linear curriculum will exceed 9,330 correct predictions by reducing late optimization noise without the full-run overhead that caused EMA verification to time out.
change: Restore static 0.02 label smoothing and uniformly average only the final 10% of learned parameter states, installing the average before validation.
mechanism: Low-overhead tail weight averaging
evidence_used: Linear ensemble annealing with static 0.02 smoothing achieved the best result of 9,330 correct; terminal smoothing decay fell to 9,322, while full-run EMA timed out, motivating a cheaper tail-only stabilization test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Linearly increasing ensemble supervision from 0.60 to 0.90 will exceed 9,330 correct predictions by preserving the beneficial temporal shift while avoiding the endpoint extremes implicated by the weaker cosine curriculum.
change: Replace the cosine 0.50-to-1.00 schedule with a same-average linear 0.60-to-0.90 schedule; retain the proven architecture, optimizer, and static 0.02 label smoothing.
mechanism: Compressed linear flip-loss curriculum
evidence_used: The linear 0.50-to-1.00 curriculum achieved 9,330 correct, versus 9,325 for static 0.75 and 9,322 for the more endpoint-concentrated cosine schedule. A compressed linear schedule isolates whether the temporal progression is beneficial without excessive early view regularization or complete late removal of view supervision.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring translation-free inputs and shortening warmup from 5% to 2% will exceed 9,330 correct predictions by providing more peak-rate optimization within the fixed exposure while preserving the winning late-training schedule.
change: Remove the harmful random translations from Reference Design 1 and shorten only its learning-rate warmup.
mechanism: Short-warmup ensemble optimization
evidence_used: Translation augmentation reduced the winning design from 9,330 to 9,222 correct, while changing the late cosine floor also regressed; this motivates restoring the winner and testing the untouched early learning-rate allocation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 82.87196491705254, "validation_accuracy": 0.9315, "validation_correct": 9315, "validation_cross_entropy": 0.20446708946228026, "validation_score": 9315.415121346507}



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
