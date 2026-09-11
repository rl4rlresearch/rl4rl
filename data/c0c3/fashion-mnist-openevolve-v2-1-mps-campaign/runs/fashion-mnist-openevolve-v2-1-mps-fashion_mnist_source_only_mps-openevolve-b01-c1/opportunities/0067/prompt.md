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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 61.38587808399461, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21609233474731446, "validation_score": 9247.411152990373}
prior_hypothesis: Ramping flip-pair allocation from 25% early to 50% late will exceed 9,246 correct predictions by preserving individual-view feature learning early and emphasizing validation-aligned flip aggregation near convergence.

## Recent verification evidence

RECENT RESULT
hypothesis: A linear 25%-to-50% ramp will exceed 9,247 correct predictions by retaining beneficial temporal allocation while reducing the excessive early/late concentration that caused the sharpened cosine curriculum to regress.
change: Replace the cosine flip-pair weight schedule with a linear ramp having identical endpoints and mean allocation.
mechanism: Linear flip-pair supervision curriculum
evidence_used: Fixed 37.5% supervision reached 9,246 correct, the cosine ramp improved to 9,247, and sharpening that ramp regressed to 9,244; a linear ramp tests an intermediate degree of temporal separation without changing architecture, compute, or average loss weighting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 56.93546687508933, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21609688491821288, "validation_score": 9247.411151451994}

RECENT RESULT
hypothesis: Allocating 25% of the full-view pair objective to the aggregate of both antithetic crop pairs will exceed 9,247 correct predictions by teaching crop-level complementarity that matches validation-time crop averaging.
change: Add cross-entropy on the four-logit antithetic crop aggregate and blend it into the existing full flip-pair loss, preserving all architecture, augmentation, schedules, and final ensemble supervision.
mechanism: Hierarchical antithetic-crop aggregation
evidence_used: Flip-pair hierarchy improved correctness from 9,239 to 9,245 and its cosine curriculum reached 9,247, while further schedule-shape refinements did not improve; this motivates extending the successful hierarchy to the currently unsupervised intermediate aggregation across full-crop pairs.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 52.644312290940434, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21613882446289062, "validation_score": 9240.411137273099}

RECENT RESULT
hypothesis: Independently sampling crop offsets for four groups within each batch will exceed 9,247 correct predictions by increasing stochastic crop diversity without adding model evaluations or the prohibitive overhead of per-example sampling.
change: Split each padded batch into four contiguous groups, assign each group independent full and central crop offsets, and preserve the existing antithetic crops, flips, losses, architecture, and optimizer.
mechanism: Random crop microbatching
evidence_used: Per-example crop sampling timed out, while step-level deterministic offset balancing reached only 9,238 correct; inexpensive within-batch stochastic diversity tests the remaining middle ground while retaining the 9,247-correct curriculum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 54.73716212506406, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.2160493999481201, "validation_score": 9237.411167506863}

RECENT RESULT
hypothesis: Reducing the batch size from 128 to 96 will exceed 9,247 correct predictions by providing 1,042 optimizer updates instead of 782 under the same fixed exposure and cosine curricula, improving convergence without changing the proven model or aggregation objective.
change: Set the training batch size to 96 while preserving architecture, losses, augmentation, optimizer, and progress-normalized schedules.
mechanism: Smaller-minibatch update densification
evidence_used: The 25%-to-50% cosine pair curriculum is best at 9,247 correct, while further loss-schedule and crop-sampling refinements tied or regressed; every reported verification used batch size 128 and 782 steps, leaving optimizer-update density as an informative untested axis.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing central-crop allocation from 10% to 12.5% will exceed 9,247 correct predictions by modestly favoring less-displaced views while preserving the successful full-crop ensemble.
change: Reweight validation aggregation and all corresponding training objectives from 90/10 to 87.5/12.5 full-versus-central allocation.
mechanism: Moderate central-crop emphasis
evidence_used: Deterministic crop balancing, crop microbatching, and added antithetic-crop supervision regressed to 9,238, 9,237, and 9,240 correct; refining the existing validation-aligned crop mixture tests a cheaper, narrower augmentation axis without disturbing the 9,247-correct curriculum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 57.590635458007455, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.21564779586791993, "validation_score": 9239.411303341067}

RECENT RESULT
hypothesis: Replacing the shallow fixed-coordinate decoder with bottleneck spatial refinement and coarse multi-scale pooling will exceed 9,247 correct predictions by learning part-to-whole structure with smoother translation behavior.
change: Add two efficient residual bottleneck blocks and replace the flattened 7×7 head with a 1×1/2×2/3×3 spatial-pyramid descriptor and wider classifier, totaling 248,402 learned parameters.
mechanism: Residual spatial-pyramid classifier
evidence_used: Widening the flattened head regressed to 9,210 and appended global summaries reached only 9,236, indicating that capacity added within the fixed-coordinate mechanism was insufficient; this patch instead reallocates capacity to spatial feature computation and structured aggregation while avoiding the timed-out attention-pooling mechanism.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Supplying pooled intensity and local-maximum maps directly to the successful fixed-coordinate decoder will exceed 9,247 correct predictions by restoring silhouette and texture information lost through two max-pooling stages.
change: Concatenate parameter-free 7×7 average- and max-pooled image maps with the learned 48-channel feature map, expanding only the classifier input and retaining all proven training and ensemble settings.
mechanism: Localized raw-image spatial shortcut
evidence_used: Widening the flattened head regressed to 9,210 and global summaries reached only 9,236, suggesting that generic capacity and spatially collapsed statistics are insufficient; this instead adds localized low-level evidence while preserving the 9,247-correct decoder and curriculum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 238282, "training_seconds": 63.95912841707468, "validation_accuracy": 0.9224, "validation_correct": 9224, "validation_cross_entropy": 0.21796117897033693, "validation_score": 9224.410522115675}

RECENT RESULT
hypothesis: Evaluating a 0.99-decay exponential average of the optimization trajectory will exceed 9,247 correct predictions by reducing endpoint variance without changing examples, augmentation, architecture, or training time materially.
change: Maintain an exponential moving average of every learned parameter after each optimizer step and copy the averaged parameters into the model after the final step.
mechanism: Exponential moving-average checkpoint ensembling
evidence_used: Loss-schedule refinements and added architectural features tied or regressed from the 9,247-correct design, while parameter averaging remains an untested temporal-ensemble axis that preserves the proven training objective.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 60.523177166003734, "validation_accuracy": 0.9225, "validation_correct": 9225, "validation_cross_entropy": 0.21978302307128905, "validation_score": 9225.40990896786}

RECENT RESULT
hypothesis: Decreasing central-crop allocation from 10% to 7.5% will exceed 9,247 correct predictions by favoring the broader crop distribution after increasing central allocation to 12.5% regressed to 9,239.
change: Reweight validation aggregation and all corresponding training objectives from 90/10 to 92.5/7.5 full-versus-central allocation.
mechanism: Reduced central-crop emphasis
evidence_used: Raising central-crop allocation from 10% to 12.5% reduced validation correctness by eight images, providing directional evidence that the successful ensemble may benefit from less central-crop emphasis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 55.809055167017505, "validation_accuracy": 0.9238, "validation_correct": 9238, "validation_cross_entropy": 0.21665183029174806, "validation_score": 9238.41096391552}

RECENT RESULT
hypothesis: Annealing label smoothing from 0.02 to zero will exceed 9,247 correct predictions by retaining early regularization while sharpening class boundaries near convergence.
change: Apply a progress-normalized cosine decay to label smoothing in every individual, pair, and ensemble cross-entropy term, preserving all other settings.
mechanism: Cosine-decayed label smoothing
evidence_used: The cosine pair curriculum improved fixed supervision from 9,246 to 9,247 correct, showing that temporal loss allocation can help; subsequent crop, architecture, and averaging changes regressed, motivating an orthogonal target-regularization schedule.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 48.12137687508948, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.2165679069519043, "validation_score": 9243.410992265324}

RECENT RESULT
hypothesis: Adding independent ±8° rotations before the proven translation-and-flip pipeline will exceed 9,247 correct predictions by learning a missing orientation invariance without changing model capacity or view supervision.
change: Apply a vectorized random rotation to each training image before reflected padding and crop construction.
mechanism: Vectorized mild rotation augmentation
evidence_used: Schedule refinements plateaued at or below 9,247 correct, while additional translation microbatching regressed to 9,237; this motivates testing an orthogonal geometric augmentation while preserving the successful crop ensemble and loss curriculum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 53.24910895782523, "validation_accuracy": 0.9162, "validation_correct": 9162, "validation_cross_entropy": 0.23259494018554688, "validation_score": 9162.405648265865}

RECENT RESULT
hypothesis: Adding one parameter-efficient residual block at 7×7 resolution will exceed 9,247 correct predictions by improving part-to-whole feature composition while preserving the proven fixed-coordinate decoder and training objective.
change: Insert a 48→32→32→48 bottleneck residual block after the existing residual stage, increasing learned parameters from 230,442 to 242,954.
mechanism: Bottleneck spatial residual refinement
evidence_used: Widening the flattened decoder regressed to 9,210 and raw-image shortcuts reached 9,224, suggesting additional decoder capacity or low-level inputs are less useful than further learned spatial feature processing; the broader residual-pyramid attempt was not verifiable, so this isolates a conservative feature-refinement change below the parameter ceiling.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 242954, "training_seconds": 72.68157358304597, "validation_accuracy": 0.9204, "validation_correct": 9204, "validation_cross_entropy": 0.21752637176513673, "validation_score": 9204.410668722745}



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
