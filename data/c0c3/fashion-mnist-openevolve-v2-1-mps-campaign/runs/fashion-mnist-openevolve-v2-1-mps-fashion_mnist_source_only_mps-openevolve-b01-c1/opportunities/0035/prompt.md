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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 55.51774841709994, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21660788536071776, "validation_score": 9237.410978759892}
prior_hypothesis: Linearly increasing ensemble supervision from 0.60 to 0.90 while preserving an average weight of 0.75 will exceed 9,236 correct predictions by prioritizing stable per-view feature learning early and deployed-ensemble alignment late.

## Recent verification evidence

RECENT RESULT
hypothesis: Jointly supervising complementary translated views and their mirrors, including loss on their four-logit average, will exceed 9,124 correct predictions by extending the successful flip-aligned training to the translation ensemble used at evaluation.
change: Keep training batches unmodified, then construct two complementary random translations and both horizontal orientations inside the loss; blend per-view cross-entropy with four-view ensemble cross-entropy.
mechanism: Antithetic translation-and-flip ensemble supervision
evidence_used: Paired-view ensemble-aligned supervision raised validation correctness from 9,063 to 9,124, while evaluation averages translated and mirrored views; this directly tests whether aligning the remaining translation dimension yields another accuracy gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 47.484982250025496, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.2223772087097168, "validation_score": 9209.409039040025}

RECENT RESULT
hypothesis: Adding a randomly sampled central translation and its mirror with the evaluation ensemble’s exact 90% full-range/10% central weighting will exceed 9,209 correct predictions by aligning training with both components of the successful validation ensemble.
change: Extend four-view antithetic supervision to six views, using four full-range translated/mirrored logits for 90% of each loss and two central translated/mirrored logits for 10%.
mechanism: Evaluation-distribution Monte Carlo supervision
evidence_used: Four-view translation-and-flip ensemble supervision improved correctness from 9,124 to 9,209, while evaluation additionally gives 10% weight to a central-translation ensemble that the current training objective does not explicitly model.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 73.49385941703804, "validation_accuracy": 0.9218, "validation_correct": 9218, "validation_cross_entropy": 0.2183428237915039, "validation_score": 9218.410393519982}

RECENT RESULT
hypothesis: Supervising complementary central translations and their mirrors will exceed 9,218 correct predictions by reducing Monte Carlo mismatch with the validation ensemble’s nine-offset central component.
change: Expand the central training sample from one translated flip pair to two complementary translated flip pairs, preserving the evaluation-aligned 90% full-range and 10% central weighting.
mechanism: Antithetic central-translation ensemble supervision
evidence_used: Adding one randomly sampled central translation and its mirror improved correctness from 9,209 to 9,218; the full-range component already benefits from complementary translations, while the central component does not.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 58.68686404195614, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.2183436798095703, "validation_score": 9209.410393231636}

RECENT RESULT
hypothesis: Increasing the ensemble-loss weight from 0.5 to 0.75 will exceed 9,218 correct predictions by concentrating optimization on the averaged logits used at validation while retaining enough per-view supervision for stable features.
change: Reweight the existing six-view objective to 25% individual-view cross-entropy and 75% evaluation-aligned ensemble cross-entropy.
mechanism: Stronger deployed-ensemble supervision
evidence_used: Ensemble-aligned supervision produced successive gains from 9,063 to 9,124, 9,209, and 9,218 correct; expanding central sampling did not improve further, motivating stronger use of the already successful ensemble objective instead of adding views.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 44.44890237506479, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.21701934700012207, "validation_score": 9236.410839812228}

RECENT RESULT
hypothesis: Increasing the ensemble-loss weight to 0.875 while retaining 0.125 individual-view supervision will exceed 9,236 correct predictions by further aligning gradients with validation-time averaging without discarding per-view regularization.
change: Reweight the existing six-view objective from 25% individual and 75% ensemble loss to 12.5% individual and 87.5% ensemble loss.
mechanism: Higher-weight deployed-ensemble supervision
evidence_used: Raising the ensemble weight from 0.5 to 0.75 improved validation correctness from 9,218 to 9,236, the strongest current result, indicating that further emphasis on the deployed ensemble is the most promising next test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 61.542903708992526, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.21639295654296875, "validation_score": 9232.411051377198}

RECENT RESULT
hypothesis: An ensemble-loss weight of 0.80 will exceed 9,236 correct predictions by increasing validation-aligned supervision modestly without the accuracy loss observed at 0.875.
change: Reweight the six-view objective from 25% individual and 75% ensemble loss to 20% individual and 80% ensemble loss.
mechanism: Intermediate deployed-ensemble loss weighting
evidence_used: Raising the ensemble weight from 0.50 to 0.75 improved correctness from 9,218 to 9,236, while raising it further to 0.875 reduced correctness to 9,232; testing 0.80 targets the supported interval near the observed optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 55.348163874819875, "validation_accuracy": 0.923, "validation_correct": 9230, "validation_cross_entropy": 0.21665841789245605, "validation_score": 9230.410961690353}

RECENT RESULT
hypothesis: An ensemble-loss weight of 0.70 will exceed 9,236 correct predictions by preserving stronger individual-view regularization than 0.75 while retaining most of the gain from ensemble-aligned supervision.
change: Reweight the six-view objective from 25% individual and 75% ensemble loss to 30% individual and 70% ensemble loss.
mechanism: Bracketed ensemble-loss weight refinement
evidence_used: Increasing the ensemble weight from 0.50 to 0.75 raised correctness from 9,218 to 9,236, but further increases to 0.80 and 0.875 reduced it to 9,230 and 9,232; testing 0.70 probes the untested side of the apparent optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 59.82430004095659, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.2174182384490967, "validation_score": 9232.410705199092}

RECENT RESULT
hypothesis: Replacing the coordinate-specific flattened head with a deeper residual backbone and content-adaptive global-plus-part tokens will exceed 9,236 correct predictions by learning discriminative regions while making predictions intrinsically less dependent on absolute position.
change: Reallocate the flattened head’s 189,210 parameters into grouped and bottleneck residual feature extraction, then classify one global token and three learned attention-pooled semantic part tokens; preserve the best six-view loss and optimizer to isolate the representational change. The resulting model has 234,357 learned parameters.
mechanism: Learned semantic part-token pooling
evidence_used: The 0.75 ensemble-loss design reached 9,236 correct, while nearby weights of 0.70, 0.80, and 0.875 all regressed, indicating that loss-weight refinement has saturated. The best model still spends most of its capacity on an absolute-position flattened head, so adaptive part localization tests a distinct mechanism rather than another ensemble calibration.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Averaging the final half of the AdamW trajectory with decay 0.99 will exceed 9,236 correct predictions by reducing parameter noise while preserving the proven architecture and six-view objective.
change: Replace AdamW with an interface-compatible optimizer that maintains an EMA after the midpoint of training and installs the averaged parameters on the final step.
mechanism: Late-training exponential weight averaging
evidence_used: The 0.75 ensemble-loss design achieved 9,236 correct, while nearby loss weights regressed and the larger architectural replacement timed out; inexpensive weight averaging tests an orthogonal generalization improvement without increasing learned parameters or training forwards.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 49.8437739580404, "validation_accuracy": 0.9216, "validation_correct": 9216, "validation_cross_entropy": 0.2209544494628906, "validation_score": 9216.409515686863}

RECENT RESULT
hypothesis: Linearly increasing ensemble supervision from 0.60 to 0.90 while preserving an average weight of 0.75 will exceed 9,236 correct predictions by prioritizing stable per-view feature learning early and deployed-ensemble alignment late.
change: Replace the static 75% ensemble-loss weight with a training-progress curriculum from 60% to 90%, leaving the architecture, views, optimizer, and average loss weighting unchanged.
mechanism: Mean-preserving ensemble-loss curriculum
evidence_used: Static 0.75 achieved the best result at 9,236 correct, while both lower 0.70 and higher 0.80/0.875 weights regressed; preserving the successful mean while changing only its temporal allocation directly tests whether those competing supervision needs are stage-dependent.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 55.51774841709994, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21660788536071776, "validation_score": 9237.410978759892}

RECENT RESULT
hypothesis: Increasing the curriculum range from 0.60–0.90 to 0.55–0.95 will exceed 9,237 correct predictions by strengthening early per-view feature learning and late deployed-ensemble alignment while preserving the successful average ensemble weight of 0.75.
change: Steepen the linear ensemble-supervision curriculum without changing its mean, architecture, views, optimizer, or evaluation.
mechanism: Stronger mean-preserving ensemble-loss curriculum
evidence_used: Static 0.75 produced 9,236 correct, while the mean-preserving 0.60–0.90 curriculum improved this to 9,237; widening that curriculum directly tests whether greater stage specialization extends the observed gain.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 59.04651687480509, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.21623482475280761, "validation_score": 9235.411104821063}

RECENT RESULT
hypothesis: A 0.65–0.85 curriculum will exceed 9,237 correct predictions by retaining beneficial stage-dependent supervision while avoiding the excessive specialization observed with the wider 0.55–0.95 range.
change: Narrow the linear ensemble-loss curriculum from 0.60–0.90 to 0.65–0.85 while preserving its mean weight of 0.75.
mechanism: Narrower mean-preserving ensemble-loss curriculum
evidence_used: Static 0.75 reached 9,236 correct, the 0.60–0.90 curriculum improved to 9,237, and widening it to 0.55–0.95 regressed to 9,235; this motivates testing an intermediate curriculum amplitude.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 49.21014712494798, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.21673590965270997, "validation_score": 9236.41093551693}



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
