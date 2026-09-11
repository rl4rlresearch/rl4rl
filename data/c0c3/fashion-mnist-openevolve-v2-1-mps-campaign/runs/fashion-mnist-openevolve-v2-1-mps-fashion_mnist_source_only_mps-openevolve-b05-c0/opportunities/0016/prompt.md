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
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 82.45255679101683, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.2431012722015381, "validation_score": 9250.402219844176}
prior_hypothesis: Reducing batch size from 64 to 32 will exceed 9,235 correct predictions by increasing optimizer steps from 1,564 to approximately 3,126 within the fixed exposure budget.

## Recent verification evidence

RECENT RESULT
hypothesis: A 239,394-parameter three-convolution extension of the verified baseline, trained with batch size 128 for 782 optimizer steps, will exceed 8,928 correct validation predictions.
change: Widen the baseline to 24/48 channels, add one 48-channel convolution, use an 88-unit classifier, and halve the batch size while retaining the verified loss and optimizer schedule.
mechanism: Conservative width-and-depth scaling without augmentation
evidence_used: The 241,130-parameter residual model with crops, flips, and BatchNorm fell to 8,705 correct; this motivates preserving the baseline’s simple unaugmented training path while testing additional capacity and updates without those confounding changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 22.49971158290282, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.2671722816467285, "validation_score": 9168.39457933798}

RECENT RESULT
hypothesis: Preserving the verified training path while averaging original and horizontally flipped validation predictions will exceed 9,168 correct predictions.
change: Leave training unchanged and ensemble original-image and flipped-image class probabilities only in evaluation mode.
mechanism: Probability-space horizontal-reflection test-time ensembling
evidence_used: The current unaugmented CNN achieved 9,168 correct, while the residual model combining architectural and augmentation changes regressed to 8,705; this isolates a parameter-free symmetry improvement without disturbing the successful optimization path.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 18.34922662517056, "validation_accuracy": 0.8889, "validation_correct": 8889, "validation_cross_entropy": 0.4421930076599121, "validation_score": 8889.346694233951}

RECENT RESULT
hypothesis: Averaging the successful model’s final half-training trajectory will exceed 9,168 correct predictions by reducing optimizer noise without introducing the harmful input transformations seen in flip ensembling.
change: Maintain a 0.99-decay EMA of model parameters during the latter half of training and install the averaged weights after the final optimizer step.
mechanism: Tail exponential weight averaging
evidence_used: The unmodified training path reached 9,168 correct, while horizontal-flip evaluation fell to 8,889; weight-space averaging preserves the validated input distribution while providing parameter-free ensembling.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 21.138614665949717, "validation_accuracy": 0.917, "validation_correct": 9170, "validation_cross_entropy": 0.2676764091491699, "validation_score": 9170.394422422309}

RECENT RESULT
hypothesis: Removing 0.05 label smoothing will exceed 9,170 correct predictions by strengthening the correct-class gradient during the fixed two-pass exposure budget.
change: Retain the successful architecture, optimizer schedule, and tail EMA while changing training to ordinary cross-entropy.
mechanism: Hard-target cross-entropy for faster finite-budget fitting
evidence_used: The current unaugmented EMA design is best at 9,170 correct, while input transformations sharply regressed; this isolates a loss-level change, and the short 100,000-example budget makes label smoothing’s weaker target signal a plausible source of underfitting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 22.902583167189732, "validation_accuracy": 0.9135, "validation_correct": 9135, "validation_cross_entropy": 0.23318508911132813, "validation_score": 9135.40545414019}

RECENT RESULT
hypothesis: Increasing label smoothing from 0.05 to 0.10 will exceed 9,170 correct predictions by improving class-boundary generalization while preserving the successful architecture, schedule, and tail EMA.
change: Change only the cross-entropy label-smoothing strength from 0.05 to 0.10.
mechanism: Stronger target-distribution regularization
evidence_used: Removing label smoothing reduced validation accuracy from 9,170 to 9,135 correct despite lowering cross-entropy, indicating that regularization—not stronger finite-budget fitting—is more important for the primary accuracy objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239394, "training_seconds": 19.899079624796286, "validation_accuracy": 0.9161, "validation_correct": 9161, "validation_cross_entropy": 0.3148568885803223, "validation_score": 9161.380269521605}

RECENT RESULT
hypothesis: Adding BatchNorm after each convolution will exceed 9,170 correct predictions by improving optimization during the fixed 782-step budget while preserving the successful architecture, inputs, loss, schedule, and EMA.
change: Insert BatchNorm2d between each convolution and GELU, increasing learned parameters from 239,394 to 239,634.
mechanism: Per-layer convolutional activation normalization
evidence_used: The unaugmented three-convolution model reached 9,170 correct, whereas the prior BatchNorm residual model was confounded by a different architecture plus harmful crops and flips; isolating normalization is the most informative next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 239634, "training_seconds": 23.710664124926552, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.25434426040649416, "validation_score": 9210.398614651322}

RECENT RESULT
hypothesis: Reducing batch size from 128 to 64 will exceed 9,210 correct predictions by providing roughly twice as many optimizer updates within the fixed 100,000-example budget.
change: Change only the training batch size, preserving the successful BatchNorm architecture, loss, cosine schedule, and tail EMA.
mechanism: Higher update density from smaller minibatches
evidence_used: Moving from the original 392-update regime to the 782-update, batch-128 regime accompanied the large improvement from 8,928 to 9,168 correct; after BatchNorm raised this to 9,210, update density remains the clearest unisolated optimization variable.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 239634, "training_seconds": 44.811550999991596, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.2482459259033203, "validation_score": 9235.400562092473}

RECENT RESULT
hypothesis: Reducing batch size from 64 to 32 will exceed 9,235 correct predictions by increasing optimizer steps from 1,564 to approximately 3,126 within the fixed exposure budget.
change: Change only the training batch size, preserving the successful BatchNorm architecture, label-smoothed loss, cosine schedule, and tail EMA.
mechanism: Higher update density from smaller minibatches
evidence_used: Reducing batch size from 128 to 64 increased validation correct from 9,210 to 9,235 while holding the rest of the implementation fixed, directly motivating the next isolated reduction.
result: improved the objective and became an available design
reported_values: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 239634, "training_seconds": 82.45255679101683, "validation_accuracy": 0.925, "validation_correct": 9250, "validation_cross_entropy": 0.2431012722015381, "validation_score": 9250.402219844176}

RECENT RESULT
hypothesis: Reducing batch size from 32 to 16 will exceed 9,250 correct predictions by increasing optimizer steps from 3,126 to approximately 6,250 while preserving the proven model and training procedure.
change: Change only the training batch size from 32 to 16.
mechanism: Higher update density from smaller minibatches
evidence_used: Successive isolated reductions from batch size 128 to 64 to 32 improved validation correct from 9,210 to 9,235 to 9,250, directly motivating one further reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing batch size from 32 to 24 will exceed 9,250 correct predictions by increasing optimizer steps to approximately 4,168 while still completing within the verification time limit.
change: Change only the training batch size from 32 to 24.
mechanism: Intermediate minibatch update density
evidence_used: Isolated reductions from batch sizes 128 to 64 to 32 improved validation correct from 9,210 to 9,235 to 9,250, while batch size 16 timed out; batch size 24 tests a higher update density at an intermediate runtime cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the EMA update rate from 0.01 to 0.0025 will exceed 9,250 correct predictions by restoring the same examples-based averaging horizon that benefited the batch-128 model.
change: Change only the EMA interpolation rate to account for the fourfold increase in optimizer steps caused by reducing batch size from 128 to 32.
mechanism: Exposure-normalized tail weight averaging
evidence_used: Tail EMA improved the batch-128 design from 9,168 to 9,170 correct; at batch 32, the unchanged per-step decay averages only one quarter as many examples, motivating a fourfold lower interpolation rate.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Keeping 10% of the peak learning rate at the end of training will exceed 9,250 correct predictions by making the final optimizer updates productive while tail EMA controls their noise.
change: Change the cosine schedule from decay-to-zero to decay-to-10%-of-peak, leaving the proven batch-32 model, loss, optimizer, and EMA unchanged.
mechanism: Nonzero cosine learning-rate floor
evidence_used: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that additional effective updates help; smaller batches timed out, so increasing the usefulness of existing late updates is the next runtime-neutral test.
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
