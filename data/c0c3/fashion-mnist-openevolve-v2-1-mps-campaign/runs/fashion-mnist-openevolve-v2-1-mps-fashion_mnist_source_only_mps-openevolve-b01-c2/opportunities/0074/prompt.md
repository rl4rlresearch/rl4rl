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
hypothesis: Scaling ensemble logits to 1.175 will preserve exactly 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.15 to 1.175.
mechanism: Conservative evaluation-temperature sharpening
evidence_used: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while cross-entropy decreased from 0.200571 to 0.198018 to 0.196639; 1.175 cautiously approaches the fitted optimum near 1.184, and prior higher-scale timeouts provide no negative validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing label smoothing from 0.02 to 0.025 will exceed 9,330 correct predictions without increasing runtime, while retaining the verified 1.15 evaluation calibration.
change: Raise smoothing for both ensemble and per-view cross-entropy losses to 0.025; preserve all other architecture, optimization, and inference settings.
mechanism: Moderately stronger fixed target smoothing
evidence_used: Annealing smoothing toward zero reduced correctness from 9,330 to 9,312, supporting soft targets for accuracy; the 0.03 trial timed out without negative model evidence, so 0.025 tests the indicated direction more conservatively.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.05 to 1.15 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196639.
change: Change only the evaluation-time symmetric flip-ensemble logit scale to 1.15.
mechanism: Verified symmetric-ensemble temperature sharpening
evidence_used: All three qualified reference designs use the otherwise identical implementation and achieved 9,330 correct with 0.196639 cross-entropy at scale 1.15, strictly outperforming the current scale 1.05; later timeouts provide no contrary validation result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Narrowing the residual refinement’s hidden channels from 64 to 48 while widening the classifier bottleneck from 35 to 40 will exceed 9,330 correct predictions, remain below 250,000 parameters, and reduce computation slightly.
change: Reallocate parameters from the full-width residual branch to the spatial classifier while preserving all verified training, augmentation, scheduling, and inference settings.
mechanism: Dense residual-bottleneck capacity reallocation
evidence_used: The current dense architecture and training procedure reliably achieved 9,330 correct; the prior grouped-convolution reallocation was not validated because it timed out, motivating a simpler dense bottleneck that increases classifier capacity without adding operators.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Cycling through centered and four one-pixel translations will exceed 9,330 correct predictions by improving shift invariance without adding parameters or model-forward work.
change: Apply centered, up, down, left, and right training views using a single contiguous `torch.roll` operation per augmented batch.
mechanism: Low-overhead deterministic translation augmentation
evidence_used: The verified flip-invariant procedure achieved 9,330 correct; the prior translation experiment timed out without producing negative accuracy evidence, motivating a lower-overhead implementation of the same spatial-invariance idea.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing the dense refinement with a 1×1–3×3–1×1 bottleneck and widening the classifier to 52 units will exceed 9,330 correct predictions while reducing convolutional work enough to finish verification, with 248,750 learned parameters.
change: Compress the residual refinement through 40 channels, add an intermediate nonlinearity, and reallocate the saved parameters to the spatial classifier.
mechanism: Compute-efficient residual bottleneck capacity reallocation
evidence_used: The verified model reached 9,330 correct but required 80.73 seconds, while many equally or more expensive variants timed out; the 48-channel/40-unit reallocation also timed out, motivating a substantially cheaper bottleneck that preserves spatial refinement and supports a wider classifier.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.05 to 1.15 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196639.
change: Change only the evaluation-time symmetric flip-ensemble logit scale to 1.15.
mechanism: Verified symmetric-ensemble temperature sharpening
evidence_used: All three qualified reference designs achieved 9,330 correct with 0.196639 cross-entropy using scale 1.15, strictly outperforming the current scale 1.05 at identical correctness.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Evaluating both flip views in one batch and scaling their mean by 1.184 will preserve 9,330 correct predictions, reduce cross-entropy below 0.196639, and lower evaluation overhead.
change: Replace two sequential evaluation forwards with one concatenated forward and use the quadratic-fit calibration scale of 1.184.
mechanism: Batched flip ensembling with fitted logit calibration
evidence_used: Scales 1.05, 1.10, and 1.15 retained 9,330 correct while cross-entropy decreased toward a fitted optimum near 1.184; the direct 1.184 attempt timed out, motivating equivalent batched evaluation with less dispatch overhead.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding mild symmetric KL agreement between original and flipped predictions will exceed 9,330 correct by strengthening the flip invariance already responsible for the best verified result.
change: Add a parameter-free, progressively weighted consistency term to the existing ensemble and per-view loss without changing forwards, architecture, or evaluation.
mechanism: Ramp-weighted bidirectional flip consistency
evidence_used: Symmetric flip training and inference achieved the best verified 9,330 correct; this directly reinforces that proven symmetry while avoiding the extra image operations associated with timed-out translation experiments.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling the flip-ensemble logits to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy below 0.196639.
change: Change only the positive evaluation-time logit scale from 1.15 to the fitted cross-entropy optimum of 1.184.
mechanism: Quadratic-fit evaluation logit calibration
evidence_used: Verified scales 1.05, 1.10, and 1.15 preserved 9,330 correct while cross-entropy decreased toward a fitted optimum near 1.184; the prior 1.184 attempt timed out and provides no contrary validation evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Scaling evaluation logits from 1.05 to 1.15 will preserve all 9,330 correct predictions while reducing validation cross-entropy to approximately 0.196639.
change: Change only the evaluation-time symmetric flip-ensemble logit scale to 1.15.
mechanism: Verified flip-ensemble temperature sharpening
evidence_used: All three qualified reference designs use this exact change and achieved 9,330 correct with 0.196639 cross-entropy, outperforming the current 0.200571 cross-entropy at identical correctness; subsequent attempts yielded no contrary validation result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling evaluation logits to 1.18 will preserve exactly 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.
change: Increase only the symmetric flip-ensemble evaluation scale from 1.15 to 1.18.
mechanism: Near-optimal evaluation-temperature sharpening
evidence_used: Verified scales 1.05, 1.10, and 1.15 all retained 9,330 correct while cross-entropy decreased toward a fitted optimum near 1.184; positive logit scaling preserves argmax, and prior higher-scale attempts timed out without contrary validation evidence.
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
