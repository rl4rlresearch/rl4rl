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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 44.44890237506479, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.21701934700012207, "validation_score": 9236.410839812228}
prior_hypothesis: Increasing the ensemble-loss weight from 0.5 to 0.75 will exceed 9,218 correct predictions by concentrating optimization on the averaged logits used at validation while retaining enough per-view supervision for stable features.

## Recent verification evidence

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.30 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255522.
change: Increase the evaluation-only logit scale from 1.25 to 1.30 without changing training, parameters, or class decisions.
mechanism: Decision-preserving near-optimal logit sharpening
evidence_used: Every scale increase from 1.05 through 1.25 preserved all 9,063 correct predictions and reduced cross-entropy; the latest increase improved it from 0.256638 to 0.255522, though the shrinking gains indicate the calibration optimum is approaching.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 46.587361083133146, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2551949073791504, "validation_score": 9063.398344509733}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.295 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255195.
change: Reduce the decision-preserving evaluation-only logit scale from 1.30 to 1.295.
mechanism: Quadratic-guided logit temperature calibration
evidence_used: Cross-entropy improvements shrank from 0.001116 at 1.20→1.25 to 0.000327 at 1.25→1.30; quadratic interpolation of those measurements places the calibration minimum near 1.296.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 38.9162491671741, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2551960182189941, "validation_score": 9063.398344157202}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.298 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255194907.
change: Reduce the decision-preserving evaluation-only logit scale from 1.30 to 1.298.
mechanism: Local quadratic logit-temperature calibration
evidence_used: The nearby 1.295 and 1.30 results retained identical predictions, with cross-entropies of 0.255196018 and 0.255194907; fitting these with the 1.25 result places the local minimum near 1.2983.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 28.985292833996937, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.2551945587158203, "validation_score": 9063.398344620384}

RECENT RESULT
hypothesis: Reallocating capacity from the coordinate-specific flattened head into deeper residual feature extraction and coarse spatial pooling will achieve at least 9,064 correct predictions under the same exposure budget.
change: Replace the shallow CNN and 50-view absolute-position compensation with a 243,042-parameter residual CNN whose classifier combines global average, quadrant-average, and global-max evidence; use an 18-view central translation/flip ensemble.
mechanism: Coarse-layout spatial-pyramid residual CNN
evidence_used: The current design devotes 189,210 of 230,442 parameters to its flattened head, while extensive ensemble calibration has plateaued at exactly 9,063 correct predictions. This tests whether learned translation-tolerant shape features can outperform further post-hoc refinement of the same decisions.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.29834 will preserve exactly 9,063 correct predictions while lowering validation cross-entropy below 0.255194559.
change: Increase the decision-preserving evaluation-only logit scale from 1.298 to 1.29834.
mechanism: Local quadratic logit-temperature refinement
evidence_used: The 1.295, 1.298, and 1.300 scales produced cross-entropies of 0.255196018, 0.255194559, and 0.255194907 with identical predictions; quadratic interpolation places the minimum near 1.29834.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 31.243253959110007, "validation_accuracy": 0.9063, "validation_correct": 9063, "validation_cross_entropy": 0.25519453887939453, "validation_score": 9063.39834462668}

RECENT RESULT
hypothesis: Training jointly on every image and its mirror, including loss on their averaged logits, will exceed 9,063 correct predictions by aligning optimization with the fixed flip-ensemble evaluation.
change: Replace single-view cross-entropy with an equal blend of per-view and flip-averaged cross-entropy.
mechanism: Paired-view ensemble-aligned supervision
evidence_used: Evaluation-only temperature tuning has plateaued at 9,063 correct, while inference averages every translated view with its horizontal flip; the current training loss supervises only one randomly selected orientation per example.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 35.848306082887575, "validation_accuracy": 0.9124, "validation_correct": 9124, "validation_cross_entropy": 0.2371648178100586, "validation_score": 9124.404149869768}

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
