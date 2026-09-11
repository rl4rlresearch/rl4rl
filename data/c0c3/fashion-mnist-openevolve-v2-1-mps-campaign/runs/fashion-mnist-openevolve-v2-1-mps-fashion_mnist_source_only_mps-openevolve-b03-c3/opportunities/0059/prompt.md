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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.30423016613349, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758353462219238, "validation_score": 9240.410649442754}
prior_hypothesis: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 39.156726666959, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2169271728515625, "validation_score": 9240.410870930615}
prior_hypothesis: Annealing label smoothing from 0.04 to zero will exceed 9,240 correct predictions by strengthening class separation during the final low-learning-rate updates.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 31.942004709038883, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}
prior_hypothesis: Scaling evaluation logits from 1.25 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2178642181 to approximately 0.2175833054.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.103174792137, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758494262695313, "validation_score": 9240.410648967883}
prior_hypothesis: Increasing evaluation-logit scaling from 1.25 to 1.30 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217864.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.295 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy to approximately 0.2175833054.
change: Replace only the evaluation-time logit multiplier with the best verified calibration.
mechanism: Verified evaluation-logit calibration
evidence_used: Reference Designs 1 and 2 achieved the best available score, 9240.410649520078, with the identical training recipe and a 1.296352 multiplier.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Changing the evaluation-only logit scale from 1.30 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2175849426 to approximately 0.2175833054.
change: Replace only the final evaluation-logit multiplier, leaving training, architecture, augmentation, and class decisions unchanged.
mechanism: Verified evaluation-logit calibration
evidence_used: Reference Designs 1 and 3 used the otherwise identical implementation at scale 1.296352 and both achieved the best available score, 9240.410649520078; architecture gating, EMA, and consistency changes all reduced validation correctness.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Annealing label smoothing from 0.04 to zero will exceed 9,240 correct predictions by strengthening class separation during the final low-learning-rate updates.
change: Reduce the average smoothing from 0.03 to 0.02 and eliminate smoothing at the end of training, while preserving the proven architecture, augmentation, optimizer, and evaluation calibration.
mechanism: Late hard-label fine-tuning
evidence_used: The baseline’s optimal 1.296352 evaluation scale indicates softened logits, while added architecture, EMA, and consistency mechanisms reduced correctness; a minimal objective refinement directly targets convergence within the fixed 522 optimizer steps.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 39.156726666959, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2169271728515625, "validation_score": 9240.410870930615}

RECENT RESULT
hypothesis: Adding content-dependent interactions among all 49 spatial locations will exceed 9,240 correct predictions by capturing relationships between distant garment parts that the locally processed dense baseline cannot model nonlinearly.
change: Replace the 48-unit dense bottleneck with a pre-normalized spatial self-attention block and a smaller 32-unit classifier, remaining under the parameter ceiling; also retain the strongest verified smoothing anneal.
mechanism: Residual nonlocal token attention
evidence_used: The baseline reached 9,240 correct, while separable coordinate gating fell to 9,225, suggesting coarse axis summaries are harmful. This tests a distinct mechanism—pairwise content-conditioned spatial mixing—and builds on the verified zero-ending smoothing improvement.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the terminal learning rate from 1e-4 to 2e-4 on top of the best verified smoothing anneal will exceed 9,240 correct predictions by giving late hard-label updates more influence.
change: Adopt the verified 1.296352 calibration and zero-ending smoothing schedule, then double the cosine schedule’s learning-rate floor without adding steps or computation.
mechanism: Stronger terminal hard-label optimization
evidence_used: Zero-ending smoothing achieved the best available score with 9,240 correct, while EMA fell to 9,226 and batch-size 128 timed out; a stronger late learning-rate tail tests additional final-iterate optimization while preserving the successful model and runtime profile.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.92078866693191, "validation_accuracy": 0.9205, "validation_correct": 9205, "validation_cross_entropy": 0.21834832763671874, "validation_score": 9205.410391666044}

RECENT RESULT
hypothesis: Lowering the terminal learning rate from 1e-4 to zero on the best verified smoothing schedule will exceed 9,240 correct predictions by suppressing harmful late hard-label updates.
change: Adopt the verified 1.296352 calibration and zero-ending label smoothing, then remove the cosine schedule’s learning-rate floor.
mechanism: Zero-terminal cosine optimization with hard-label annealing
evidence_used: Zero-ending smoothing produced the best available score with 9,240 correct, while raising the terminal learning rate to 2e-4 reduced correctness to 9,205; this directly motivates testing a lower terminal rate.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 33.60839649988338, "validation_accuracy": 0.9219, "validation_correct": 9219, "validation_cross_entropy": 0.22335164985656739, "validation_score": 9219.408713226536}

RECENT RESULT
hypothesis: Ending label smoothing halfway through training will exceed 9,240 correct predictions by preserving early regularization while allowing a full epoch of hard-label optimization.
change: Replace the cosine 0.04-to-zero smoothing schedule with a linear 0.04-to-zero anneal over the first half of training, followed by zero smoothing.
mechanism: Front-loaded label smoothing with hard-label second half
evidence_used: Reducing average smoothing from 0.03 to 0.02 lowered validation cross-entropy from 0.2175833 to 0.2169272 without losing any of the 9,240 correct predictions, motivating a further reduction in smoothing duration while leaving the proven architecture, optimizer, and learning-rate schedule unchanged.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 40.07350220903754, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.21615212669372558, "validation_score": 9233.411132776093}

RECENT RESULT
hypothesis: Preserving the baseline’s 0.05 initial smoothing while annealing to hard labels will exceed 9,240 correct predictions by combining stronger early regularization with the beneficial zero-smoothing finish.
change: Change only the cosine label-smoothing schedule from 0.05-to-0.01 to 0.05-to-zero.
mechanism: Full-horizon 0.05-to-zero label-smoothing anneal
evidence_used: The verified 0.04-to-zero schedule preserved all 9,240 correct predictions and improved cross-entropy to 0.2169272, while ending smoothing halfway reduced correctness to 9,233; this motivates retaining smoothing across the full training horizon while isolating the benefit of a zero endpoint.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.10391112510115, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21734041442871094, "validation_score": 9237.41073145529}

RECENT RESULT
hypothesis: Ending with pure flip-averaged supervision while preserving the current mean ensemble weight will exceed 9,240 correct predictions by aligning late optimization with flip-averaged evaluation without forcing view agreement.
change: Adopt the best verified smoothing and calibration, and widen the ensemble-weight schedule from 0.55→0.95 to 0.50→1.00.
mechanism: Late flip-ensemble objective alignment
evidence_used: The 0.04→0 smoothing schedule retained 9,240 correct and achieved the best cross-entropy, while explicit flip-consistency reduced correctness to 9,234; optimizing the evaluated average permits complementary views without penalizing disagreement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 40.80001358385198, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.2169219982147217, "validation_score": 9239.410872677734}

RECENT RESULT
hypothesis: Adding a zero-initialized nonlinear residual transformation to the 48-dimensional classifier representation, atop the verified 0.04→0 smoothing schedule, will exceed 9,240 correct predictions without destabilizing the successful initial optimization path.
change: Insert a lightweight residual MLP block in the classifier, adopt the best verified label-smoothing schedule, and use the verified 1.296352 evaluation-logit calibration; total learned parameters become 244,994.
mechanism: Zero-initialized residual classifier refinement
evidence_used: The 0.04→0 smoothing design retained 9,240 correct and achieved the best available cross-entropy, while spatial gating reduced correctness and attention exceeded the time limit. A zero-initialized residual head adds inexpensive global nonlinear capacity while initially preserving the proven classifier exactly.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 244994, "training_seconds": 51.77934629190713, "validation_accuracy": 0.9231, "validation_correct": 9231, "validation_cross_entropy": 0.22194619674682617, "validation_score": 9231.409183318652}

RECENT RESULT
hypothesis: Reducing initial smoothing from 0.04 to 0.03 while retaining the gradual zero endpoint will exceed 9,240 correct predictions by easing early underconfidence without prematurely removing regularization.
change: Change only the cosine label-smoothing schedule from 0.04→0 to 0.03→0.
mechanism: Lower-amplitude full-horizon label-smoothing anneal
evidence_used: The 0.04→0 schedule achieved the best score, while 0.05→0 fell to 9,237 correct; the unsuccessful half-horizon anneal indicates smoothing should still decay across the full training run.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 43.101726291002706, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.2168480842590332, "validation_score": 9236.41089763502}

RECENT RESULT
hypothesis: A 0.04→0.01 smoothing schedule will exceed 9,240 correct predictions by combining the empirically optimal 0.04 initial smoothing with modest late regularization that avoids fully hard-label updates.
change: Change only the cosine label-smoothing schedule from 0.05→0.01 to 0.04→0.01.
mechanism: Mild terminal label smoothing
evidence_used: The 0.04→0 schedule achieved the best score with 9,240 correct, while both 0.03→0 and 0.05→0 reduced correctness; the current 0.05→0.01 schedule also reached 9,240, making 0.04→0.01 the informative missing combination.
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
