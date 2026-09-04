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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 71.47688325005583, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.21195276412963868, "validation_score": 9290.412557332924}
prior_hypothesis: Reducing the arithmetic ensemble temperature from 0.95 to 0.90 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2174928.

## Recent verification evidence

RECENT RESULT
hypothesis: A power mean of order 0.5 will exceed 9,290 correct predictions by adding some consensus preference while retaining more of the arithmetic ensemble’s confident-view corrections than the geometric mean.
change: Replace arithmetic probability averaging with a compute-neutral order-0.5 power mean across translated, flipped, live, and EMA predictions.
mechanism: Intermediate power-mean view ensemble
evidence_used: Geometric averaging improved cross-entropy from 0.22418 to 0.21799 but reduced correct predictions only slightly, from 9,290 to 9,286; an intermediate aggregation directly tests whether its calibration benefit can be retained without crossing the decision boundaries lost at the geometric endpoint.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Adding a zero-initialized global mean/max classification branch will exceed 9,290 correct predictions by providing translation-invariant class evidence while preserving the validated convolutional widths and primary head.
change: Add a compute-light 1,290-parameter auxiliary classifier over global mean and maximum feature statistics, raising the model to 249,126 parameters.
mechanism: Dual-statistic global residual classifier
evidence_used: Expanding the dense head from 48 to 58 units produced the 9,290-correct best result, while sacrificing convolutional width for further expansion regressed to 9,264; the residual adapter timed out, motivating a cheaper complementary head that retains all validated channels.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249126, "training_seconds": 70.90934112505056, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.2222947067260742, "validation_score": 9281.409066649187}

RECENT RESULT
hypothesis: Increasing centered training views from 20% to 33% will exceed 9,290 correct predictions by preserving centered-image detail while retaining balanced one-pixel translation robustness.
change: Duplicate the centered offset in the augmentation choices, giving it twice the probability of each cardinal shift without changing compute or evaluation.
mechanism: Center-weighted translation augmentation
evidence_used: Removing centered training views reduced validation correct from 9,290 to 9,270, directly motivating a controlled increase in centered-view frequency while preserving all validated training components.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 64.4458514591679, "validation_accuracy": 0.9274, "validation_correct": 9274, "validation_cross_entropy": 0.22328365020751953, "validation_score": 9274.408735946005}

RECENT RESULT
hypothesis: An order-1.5 power mean will exceed 9,290 correct predictions by modestly emphasizing confident translated, flipped, live, and EMA predictions.
change: Replace arithmetic probability averaging with a compute-neutral order-1.5 power mean.
mechanism: Confidence-emphasized power-mean view ensemble
evidence_used: Geometric averaging improved cross-entropy but reduced correct predictions from 9,290 to 9,286, indicating that suppressing confident-view corrections was harmful and motivating a controlled move beyond the arithmetic mean in the opposite direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 69.83069566590711, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.22771488151550293, "validation_score": 9287.407260682043}

RECENT RESULT
hypothesis: Expanding the first convolution from 32 to 39 channels will exceed 9,290 correct predictions by increasing low-level feature diversity while preserving the validated 32/64-channel outputs and 58-unit head.
change: Widen the first convolution and its BatchNorm layer to 39 channels, adjusting the following convolution’s input width; this raises the model from 247,836 to 249,929 parameters.
mechanism: Early-stage channel expansion
evidence_used: Reducing later convolutional width to enlarge the head fell from 9,290 to 9,264 correct, while the auxiliary global head reached only 9,281; this motivates spending the remaining budget inside the feature extractor without sacrificing validated downstream capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the batch size from 64 to 50 will exceed 9,290 correct predictions by increasing optimizer updates from 1,564 to exactly 2,000 while preserving the validated model, augmentation, loss, schedule, and parameter count.
change: Use 50-example minibatches without changing any other training or evaluation behavior.
mechanism: Finer-grained fixed-exposure optimization
evidence_used: The 9,290-correct design received only 1,564 optimizer updates, while recent architectural, regularization, and ensemble changes regressed or timed out; update granularity remains untested and can increase learning opportunities without adding parameters or expensive computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Disabling the 10% head dropout will exceed 9,290 correct predictions by allowing the validated 58-unit bottleneck to learn consistently during the limited 1,564-step training run.
change: Replace the classifier’s dropout layer with an identity operation, preserving architecture, parameters, augmentation, optimizer, and evaluation behavior.
mechanism: Deterministic dense-head feature utilization
evidence_used: Expanding the dense bottleneck from 48 to 58 produced the best 9,290-correct result, indicating that head capacity is valuable; using all 58 learned features on every update may exploit that capacity better without the runtime risk of further architectural expansion.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Halving head dropout from 10% to 5% will exceed 9,290 correct predictions by exposing the validated 58-unit bottleneck more consistently during the limited 1,564-step run while retaining modest regularization.
change: Reduce the classifier-head dropout probability from 0.1 to 0.05 without changing parameters, compute structure, optimization, augmentation, or evaluation.
mechanism: Reduced stochastic head regularization
evidence_used: Expanding the dense bottleneck to 58 units produced the best 9,290-correct result, indicating that head capacity is valuable; the full dropout-removal experiment was not verifiable, so halving dropout is a conservative test of more consistent feature utilization.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 73.06048866710626, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.22293152694702148, "validation_score": 9275.408853634879}

RECENT RESULT
hypothesis: Scaling the arithmetic ensemble logits by 1/0.9 will retain the baseline’s 9,290 correct predictions while reducing validation cross-entropy below 0.2241766, thereby improving validation_score.
change: Apply temperature 0.9 to the final live/EMA multi-view ensemble without changing training, parameters, forward-pass count, or predicted classes.
mechanism: Decision-preserving post-ensemble temperature sharpening
evidence_used: Geometric aggregation reduced cross-entropy to 0.217986 but lost four correct predictions; positive temperature scaling can pursue that calibration improvement while preserving the arithmetic ensemble’s argmax decisions.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Annealing label smoothing from 0.04 to 0 while retaining approximately the baseline’s 0.02 average will exceed 9,290 correct predictions by regularizing early learning while allowing sharper class separation near training’s end.
change: Replace constant 0.02 label smoothing with a linear 0.04-to-0 schedule across the fixed training exposure.
mechanism: Linearly annealed label smoothing
evidence_used: Reducing head dropout from 0.10 to 0.05 lowered validation correct from 9,290 to 9,275, indicating that regularization remains useful; scheduling smoothing preserves its overall strength while avoiding persistent late-stage target bias.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Applying temperature 0.95 to the arithmetic ensemble will preserve its 9,290 correct predictions while lowering cross-entropy below 0.2241766, producing a strict validation_score improvement.
change: Scale the final ensemble log-probabilities by 1/0.95 without changing training, parameters, forward-pass count, or predicted classes.
mechanism: Conservative decision-preserving temperature sharpening
evidence_used: Geometric aggregation lowered cross-entropy from 0.2241766 to 0.217986, suggesting the baseline ensemble benefits from sharper consensus; the prior temperature-0.9 attempt timed out and therefore did not test this mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 75.05371608398855, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.2174928207397461, "validation_score": 9290.410680039737}

RECENT RESULT
hypothesis: Reducing the arithmetic ensemble temperature from 0.95 to 0.90 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2174928.
change: Scale the final ensemble log-probabilities by 1/0.90 instead of 1/0.95, with no training, parameter, or inference-pass changes.
mechanism: Stronger decision-preserving temperature sharpening
evidence_used: Temperature 0.95 preserved 9,290 correct predictions and reduced cross-entropy from 0.2241766 to 0.2174928; because positive scaling cannot change argmax decisions, 0.90 directly tests whether further sharpening improves the tie-break objective. Its earlier verification timed out, so the mechanism remains unevaluated.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247836, "training_seconds": 71.47688325005583, "validation_accuracy": 0.929, "validation_correct": 9290, "validation_cross_entropy": 0.21195276412963868, "validation_score": 9290.412557332924}



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
