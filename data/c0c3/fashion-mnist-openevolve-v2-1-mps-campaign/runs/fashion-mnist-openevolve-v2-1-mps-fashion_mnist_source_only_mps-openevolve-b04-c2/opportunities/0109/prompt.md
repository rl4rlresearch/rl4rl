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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 55.120978333987296, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20288349685668947, "validation_score": 9295.415667852545}
prior_hypothesis: BatchNorm1d momentum 0.2 will exceed 9,288 correct predictions by tracking the rapidly evolving classifier representation more closely than the best verified default momentum of 0.1.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 79.58396562491544, "validation_accuracy": 0.9292, "validation_correct": 9292, "validation_cross_entropy": 0.20335481338500977, "validation_score": 9292.415505048419}
prior_hypothesis: BatchNorm1d momentum 0.25 will exceed 9,295 correct predictions by balancing the superior adaptation at momentum 0.2 against the slight degradation observed at 0.3.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 59.471149208024144, "validation_accuracy": 0.9292, "validation_correct": 9292, "validation_cross_entropy": 0.20390634841918945, "validation_score": 9292.415314696742}
prior_hypothesis: BatchNorm1d momentum 0.3 will exceed 9,295 correct predictions by tracking the final classifier representation more closely than momentum 0.2.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 53.33525920799002, "validation_accuracy": 0.9293, "validation_correct": 9293, "validation_cross_entropy": 0.2027974105834961, "validation_score": 9293.415697602606}
prior_hypothesis: BatchNorm1d momentum 0.19 will exceed 9,295 correct predictions by retaining the rapid adaptation seen at momentum 0.2 while slightly reducing running-statistic noise.

## Recent verification evidence

RECENT RESULT
hypothesis: Flipping half of each second-exposure batch will exceed 9,281 correct predictions by improving left-right invariance while avoiding the runtime cost of padded translation augmentation.
change: Preserve the first exposure unchanged, then deterministically flip alternating images horizontally during the second exposure; retain the best architecture, smoothing, schedule, and evaluation calibration.
mechanism: Low-overhead second-pass horizontal-flip augmentation
evidence_used: Architectural channel-mixing variants underperformed the 9,281-correct baseline, while second-pass translation augmentation timed out; horizontal flipping tests the same orthogonal invariance strategy with substantially less preprocessing.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging 25% horizontally flipped logits with 75% original logits will exceed 9,281 correct predictions by adding left-right robustness without the training-time overhead that caused flip augmentation to time out.
change: Apply a parameter-free, evaluation-only weighted horizontal-flip ensemble while preserving the verified architecture, training procedure, and logit scale.
mechanism: Conservative horizontal-flip test-time ensembling
evidence_used: Training-time horizontal-flip augmentation was motivated as an orthogonal alternative after several architecture variants underperformed, but timed out before yielding accuracy evidence; evaluation-only ensembling tests that invariance without increasing training time.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 63.57808212493546, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.21591361618041993, "validation_score": 9266.41121342285}

RECENT RESULT
hypothesis: Averaging the final 64 low-learning-rate parameter states will improve validation_score beyond 9281.414347011598 by reducing endpoint noise while preserving the successful training trajectory.
change: Uniformly average the last 64 post-update model states, install the average after the final step, and use the best-verified evaluation scale of 1.20514.
mechanism: Final-window Polyak weight averaging
evidence_used: The baseline reached 9,281 correct, while broader, narrower, gated, and augmented variants failed to improve it; final-iterate averaging tests an orthogonal variance-reduction mechanism without changing the verified architecture, loss, or example exposure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 57.3805572912097, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.20648623428344726, "validation_score": 9266.414426609928}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.20514 will preserve all 9,281 correct predictions while lowering validation cross-entropy to approximately 0.2067180069.
change: Replace the current evaluation-only multiplier with the best-verified calibration scale.
mechanism: Evaluation-only logit recalibration
evidence_used: The otherwise identical reference design achieved 9,281 correct with 0.2067180069 cross-entropy at 1.20514, improving on the current 0.2068297211; positive scaling preserves argmax predictions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Normalizing the 128-unit classifier representation will exceed 9,281 correct predictions by improving optimization of the successful feature extractor while remaining under the 250,000-parameter ceiling.
change: Add BatchNorm1d between the classifier’s first linear layer and GELU; retain the verified architecture, loss, schedule, dropout, and evaluation calibration.
mechanism: Dense-representation batch normalization
evidence_used: Spatial channel-mixing, gating, depthwise refinement, augmentation, ensembling, and weight averaging all underperformed the 9,281-correct baseline; this tests a distinct, lightweight classifier-conditioning change while preserving the best feature extractor and adding only 256 learned parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 74.43629279197194, "validation_accuracy": 0.9288, "validation_correct": 9288, "validation_cross_entropy": 0.20210640487670897, "validation_score": 9288.415936557672}

RECENT RESULT
hypothesis: BatchNorm1d with the feature extractor’s verified 0.00623359375 momentum will exceed 9,288 correct predictions by retaining the successful classifier conditioning while producing less noisy evaluation statistics than the default 0.1 momentum.
change: Add BatchNorm1d after the first classifier linear layer, using the same low running-statistics momentum as the convolutional normalization layers.
mechanism: Slowly accumulated dense-feature batch normalization
evidence_used: Default-momentum dense BatchNorm achieved the best available result at 9,288 correct, while every BatchNorm2d layer in that successful feature extractor uses momentum 0.00623359375; applying that verified statistic-accumulation rate isolates whether more stable dense running statistics improve generalization.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 78.69068962498568, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20120860176086425, "validation_score": 9284.416247435514}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.05 will exceed 9,288 correct predictions by retaining faster adaptation than momentum 0.00623359375 while averaging less noisy classifier statistics than momentum 0.1.
change: Add BatchNorm1d with momentum 0.05 after the first classifier linear layer and use the best-verified evaluation scale of 1.20514.
mechanism: Intermediate-timescale dense-feature batch normalization
evidence_used: Dense BatchNorm at momentum 0.1 achieved the best result of 9,288 correct, while momentum 0.00623359375 achieved 9,284 correct but lower cross-entropy; testing an intermediate momentum directly probes the apparent adaptation-versus-statistical-noise tradeoff.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: BatchNorm1d using live validation-batch statistics will exceed 9,288 correct predictions by preserving the successful training-time normalization while eliminating the evaluation-statistics sensitivity demonstrated by momentum 0.1 versus 0.00623359375.
change: Add BatchNorm1d after the first classifier linear layer with running statistics disabled, and restore the best-verified evaluation logit scale of 1.20514.
mechanism: Validation-batch dense-feature normalization
evidence_used: Dense BatchNorm with momentum 0.1 achieved the best result at 9,288 correct, while changing only its running-statistics momentum to 0.00623359375 produced 9,284 correct; this directly motivates evaluating with current-batch statistics instead of another moving-average rate.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 54.898539332905784, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.20243156204223633, "validation_score": 9281.415824081623}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.2 will exceed 9,288 correct predictions by tracking the rapidly evolving classifier representation more closely than the best verified default momentum of 0.1.
change: Increase only the classifier BatchNorm1d momentum from its default 0.1 to 0.2 while preserving the architecture, optimization, loss, and evaluation calibration.
mechanism: Faster dense-feature running-statistic adaptation
evidence_used: Classifier BatchNorm momentum 0.1 achieved 9,288 correct, versus 9,284 at momentum 0.00623359375 and 9,281 with live validation-batch statistics; this makes faster—but still running-averaged—statistics the clearest untested direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 55.120978333987296, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20288349685668947, "validation_score": 9295.415667852545}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.3 will exceed 9,295 correct predictions by tracking the final classifier representation more closely than momentum 0.2.
change: Increase only the classifier BatchNorm1d momentum from 0.00623359375 to 0.3, preserving all other architecture, optimization, loss, and calibration settings.
mechanism: Faster dense-feature running-statistic adaptation
evidence_used: Validation correctness increased from 9,284 at momentum 0.00623359375 to 9,288 at 0.1 and 9,295 at 0.2, making a further increase the most direct test of the observed adaptation trend.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 59.471149208024144, "validation_accuracy": 0.9292, "validation_correct": 9292, "validation_cross_entropy": 0.20390634841918945, "validation_score": 9292.415314696742}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.25 will exceed 9,295 correct predictions by balancing the superior adaptation at momentum 0.2 against the slight degradation observed at 0.3.
change: Replace validation-batch statistics with running statistics at momentum 0.25, preserving all other architecture, optimization, loss, and calibration settings.
mechanism: Fine-grained dense BatchNorm momentum tuning
evidence_used: Momentum 0.2 achieved the best result of 9,295 correct, while 0.1 achieved 9,288 and 0.3 achieved 9,292; testing their unmeasured midpoint most directly refines the apparent optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 79.58396562491544, "validation_accuracy": 0.9292, "validation_correct": 9292, "validation_cross_entropy": 0.20335481338500977, "validation_score": 9292.415505048419}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19 will exceed 9,295 correct predictions by retaining the rapid adaptation seen at momentum 0.2 while slightly reducing running-statistic noise.
change: Add classifier BatchNorm1d with momentum 0.19 and use the best-verified evaluation logit scale of 1.20514.
mechanism: Fine-grained dense BatchNorm momentum refinement
evidence_used: Momentum 0.2 achieved 9,295 correct, outperforming 9,288 at 0.1, while momentum 0.25 and 0.3 fell to 9,292; this brackets the optimum near 0.2 and motivates a small downward refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 53.33525920799002, "validation_accuracy": 0.9293, "validation_correct": 9293, "validation_cross_entropy": 0.2027974105834961, "validation_score": 9293.415697602606}



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
