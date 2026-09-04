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
hypothesis: Supervising flip-averaged pairs as an intermediate aggregation level will exceed 9,239 correct predictions by aligning training with the flip-paired validation ensemble while retaining most early individual-view learning.
change: Reallocate one quarter of the individual-view loss to cross-entropy on three flip-averaged view pairs, preserving the proven six-view ensemble loss and cosine curriculum.
mechanism: Hierarchical flip-pair supervision
evidence_used: The cosine ensemble curriculum achieved the best 9,239-correct result, while architecture, head, smoothing, and EMA changes regressed; this motivates a compute-light refinement of the successful aggregation objective.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 70.5826402080711, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21621126976013183, "validation_score": 9245.411112783142}

RECENT RESULT
hypothesis: Increasing flip-pair loss allocation from 25% to 50% will exceed 9,245 correct predictions by further aligning optimization with the flip-averaged validation ensemble while retaining individual-view supervision.
change: Rebalance the intermediate view objective equally between individual-view and flip-pair cross-entropy; preserve the architecture, augmentation, six-view ensemble loss, cosine curriculum, and optimizer.
mechanism: Stronger hierarchical flip-pair supervision
evidence_used: Introducing 25% flip-pair supervision improved the best result from 9,239 to 9,245 correct, while architecture, smoothing, augmentation, and EMA changes failed to improve; increasing that successful signal is the most direct next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing fixed-coordinate flattening with six input-dependent attention pools will exceed 9,245 correct predictions by representing semantic garment parts and their spatial geometry while remaining robust to the translations already emphasized by crop ensembling.
change: Replace the flattened 7×7 classifier with six learned spatial attention heads that pool backbone channels, append global statistics and five geometric moments per head, then classify the resulting descriptor with a 440-unit head; estimated parameters are 229,710.
mechanism: Multi-head spatial part pooling with geometric moments
evidence_used: Widening the flattened head regressed to 9,210 and appending global summaries reached only 9,236, showing that additional capacity and fixed pooled statistics were insufficient; the new head instead tests input-dependent spatial aggregation while preserving the 9,245-correct backbone and hierarchical objective.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing flip-pair allocation from 25% to 37.5% will exceed 9,245 correct predictions by strengthening the aggregation signal that produced the current best result without fully committing to the unresolved 50% setting.
change: Rebalance the intermediate view loss to 62.5% individual-view and 37.5% flip-pair cross-entropy, preserving all other architecture, optimization, augmentation, and ensemble-loss settings.
mechanism: Moderate hierarchical flip-pair supervision
evidence_used: Adding 25% flip-pair supervision improved correctness from 9,239 to 9,245, while the 50% experiment timed out without accuracy evidence; testing the midpoint is the most direct dose-response follow-up.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 68.75752958306111, "validation_accuracy": 0.9246, "validation_correct": 9246, "validation_cross_entropy": 0.21604542541503907, "validation_score": 9246.411168850727}

RECENT RESULT
hypothesis: Increasing flip-pair allocation from 37.5% to 43.75% will exceed 9,246 correct predictions by continuing the observed benefit of stronger validation-aligned aggregation supervision without reaching the unresolved 50% endpoint.
change: Rebalance the view objective to 56.25% individual-view and 43.75% flip-pair cross-entropy while preserving all other settings.
mechanism: Refined hierarchical flip-pair supervision
evidence_used: Raising flip-pair allocation from 25% to 37.5% improved validation correctness from 9,245 to 9,246; testing the midpoint between 37.5% and the unverified 50% setting is the most direct dose-response follow-up.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 62.81106025003828, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.21590664138793944, "validation_score": 9242.411215781689}

RECENT RESULT
hypothesis: Ramping flip-pair allocation from 25% early to 50% late will exceed 9,246 correct predictions by preserving individual-view feature learning early and emphasizing validation-aligned flip aggregation near convergence.
change: Replace the fixed 37.5% flip-pair allocation with a cosine curriculum having the same mean allocation, while leaving the architecture, optimizer, augmentation, and ensemble-loss schedule unchanged.
mechanism: Cosine-ramped flip-pair supervision
evidence_used: Fixed flip-pair supervision improved correctness from 9,239 at 0% to 9,245 at 25% and 9,246 at 37.5%, while 43.75% regressed to 9,242; a temporal allocation tests whether stronger supervision is useful specifically after individual-view features have formed.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 61.38587808399461, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21609233474731446, "validation_score": 9247.411152990373}

RECENT RESULT
hypothesis: A steeper 25%-to-50% flip-pair curriculum will exceed 9,247 correct predictions by spending more of early training on individual-view features and concentrating validation-aligned pair supervision later, while preserving the successful endpoints and mean allocation.
change: Sharpen the existing cosine ramp with a symmetric nonlinear transform, leaving architecture, compute, optimizer, augmentation, and all other losses unchanged.
mechanism: Sharpened cosine flip-pair curriculum
evidence_used: The 25%-to-50% cosine ramp improved correctness from 9,246 to 9,247, whereas fixed 43.75% supervision regressed to 9,242; this suggests stronger pair supervision is useful when delayed rather than applied uniformly.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 46.064778625033796, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.21616091117858888, "validation_score": 9244.411129806429}

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

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
