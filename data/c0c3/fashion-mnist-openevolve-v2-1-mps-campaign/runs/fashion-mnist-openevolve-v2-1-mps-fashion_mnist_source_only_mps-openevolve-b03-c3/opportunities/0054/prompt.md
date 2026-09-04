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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 31.942004709038883, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}
prior_hypothesis: Scaling evaluation logits from 1.25 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2178642181 to approximately 0.2175833054.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 39.156726666959, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2169271728515625, "validation_score": 9240.410870930615}
prior_hypothesis: Annealing label smoothing from 0.04 to zero will exceed 9,240 correct predictions by strengthening class separation during the final low-learning-rate updates.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.30423016613349, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758353462219238, "validation_score": 9240.410649442754}
prior_hypothesis: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.103174792137, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758494262695313, "validation_score": 9240.410648967883}
prior_hypothesis: Increasing evaluation-logit scaling from 1.25 to 1.30 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217864.

## Recent verification evidence

RECENT RESULT
hypothesis: Changing the evaluation-only logit scale from 1.295 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2175835346 to approximately 0.2175833054.
change: Replace only the evaluation-logit multiplier with the best verified calibration, leaving training and class decisions unchanged.
mechanism: Verified second-order confidence calibration
evidence_used: Reference Design 1 used the identical implementation at scale 1.296352 and achieved the best available score, 9240.410649520078; architecture changes and EMA reduced validation correctness.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Changing the evaluation-only logit scale from 1.30 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy to approximately 0.2175833054.
change: Replace only the evaluation-logit multiplier, leaving training and class decisions unchanged.
mechanism: Verified evaluation-logit calibration
evidence_used: Reference Design 1 is otherwise identical and achieved the best available score, 9240.410649520078, at scale 1.296352 versus the current 9240.410648967883 at 1.30.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.25 to 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy from 0.2178642181 to approximately 0.2175833054.
change: Replace only the evaluation-time logit multiplier, leaving training and class decisions unchanged.
mechanism: Verified evaluation-logit calibration
evidence_used: Reference Design 1 used the otherwise identical implementation at scale 1.296352 and achieved the best available score, 9240.410649520078; prior architecture and EMA experiments reduced validation correctness.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 31.942004709038883, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21758330535888673, "validation_score": 9240.410649520078}

RECENT RESULT
hypothesis: Encouraging original and horizontally flipped views to agree will exceed 9,240 correct predictions by aligning training with the proven flip-averaged evaluation while preserving the successful architecture.
change: Add a gradually ramped Jensen–Shannon-style consistency loss between paired-view predictions; retain the existing supervised loss, optimizer, augmentation, and calibrated evaluation logits.
mechanism: Ramped flip-consistency regularization
evidence_used: Architecture gating and EMA reduced correctness to 9,225 and 9,226, respectively, whereas the baseline repeatedly achieved 9,240; this motivates a parameter-free change targeting the evaluation invariance already encoded by paired training and flip averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 32.66938679199666, "validation_accuracy": 0.9234, "validation_correct": 9234, "validation_cross_entropy": 0.21797940216064454, "validation_score": 9234.410515973515}

RECENT RESULT
hypothesis: Reducing the batch size to 128 will exceed 9,240 correct predictions by providing roughly 50% more optimizer updates over the fixed 100,000-example exposure while retaining the proven architecture, objective, augmentation, and evaluation ensemble.
change: Change only the training batch size from 192 to 128.
mechanism: Smaller-batch higher-frequency optimization
evidence_used: The calibrated baseline achieves 9,240 correct in only 522 optimizer steps, while added gating, EMA, and consistency regularization reduced correctness; increasing optimization frequency preserves the successful representation and tests an orthogonal route to better convergence.
result: training did not finish within the verification time limit

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
