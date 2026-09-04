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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 58.7689845408313, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.24051190643310547, "validation_score": 9232.403059412334}
prior_hypothesis: Batch-64 paired-flip training with matched cardinal crops and ten-view inference will exceed 9,119 correct predictions while remaining safely faster than the timeout-prone batch-32 designs.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 56.48495883401483, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.23800013732910155, "validation_score": 9243.403877176523}
prior_hypothesis: The verified batch-64 design with a 2% terminal learning-rate floor will complete within its demonstrated runtime headroom and exceed 9,232 correct predictions by reducing late-training parameter noise.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 47.30621508299373, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.24051190643310547, "validation_score": 9232.403059412334}
prior_hypothesis: Restoring the exact 247,538-parameter batch-64 training design will exceed the current 9,119 correct predictions and reproduce approximately 9,232 correct within the demonstrated runtime.

REFERENCE DESIGN 3
verified_results: {"batch_size": 32, "examples_processed": 100000, "optimizer_steps": 3126, "parameters": 249550, "training_seconds": 81.30000304197893, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.24333785018920898, "validation_score": 9214.402143311188}
prior_hypothesis: Replacing ±2-pixel and diagonal training crops with the exact five spatial views used by ten-view inference will exceed 9,208 correct predictions without increasing training time.

## Recent verification evidence

RECENT RESULT
hypothesis: Reference Design 3’s dense batch-32 matched-view training with a 2% terminal learning-rate floor will exceed 9,214 correct predictions by reducing late-update noise without adding runtime overhead.
change: Restore the strongest verified batch-32 flip-paired procedure, retain efficient pair-batched ten-view inference, and lower the final learning rate from 2e-4 to 4e-5.
mechanism: Compute-free near-zero cosine stabilization
evidence_used: Reference Design 3 achieved 9,214 correct, while 0.995 EMA reached 9,241 before timing out; this motivates targeting the same late-training instability through a zero-overhead lower cosine floor.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The current 96-channel network can afford Reference Design 3’s batch-32 flip-paired training and ten-view inference within the time limit, and its 3,126 matched-view updates will exceed 9,214 correct predictions.
change: Retain the current 247,538-parameter model, but use batch 32, inference-matched cardinal crops, paired mirror supervision, ten-view inference, and Reference Design 3’s learning rate.
mechanism: Width-funded dense matched-view optimization
evidence_used: Reference Design 3 achieved 9,214 correct with this training rule at 81.3 seconds; the current narrower 24/48/96 convolutional stack completed in 22.3 seconds and provides a compute-saving basis for adopting the stronger procedure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch-40 matched-view training with paired flip consistency for the first 75% of updates and a cheaper clean final quarter will finish within the time limit and exceed 9,214 correct predictions.
change: Adopt the 249,550-parameter residual model and ten-view inference, use 2,500 optimizer steps, add inexpensive flip-consistency regularization, and remove duplicated flip forwards during the final quarter.
mechanism: Front-loaded flip consistency with clean-tail fine-tuning
evidence_used: Reference Design 3 reached 9,214 correct but required 81.3 seconds; Reference Design 1 established clean late fine-tuning as effective, while the batch-40 EMA attempt timed out, motivating fewer training forwards and no parameter averaging.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training uniformly on the five inference views for all 100,000 examples will exceed 9,214 correct predictions by eliminating the current center-view overrepresentation without adding model-forward computation.
change: Extend the deterministic center-and-cardinal crop cycle through the entire training budget instead of switching to clean-only images halfway through.
mechanism: Full-exposure train–inference view matching
evidence_used: The current matched-view design achieved 9,214 correct, while prior evidence found shifted ten-view inference beneficial. Currently 60% of training examples are centered although inference weights all five spatial views equally; this patch makes both distributions 20% per view.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch 48 with 2,084 dense updates and exact cardinal-view augmentation will finish within the runtime limit and exceed the current 9,119 correct predictions.
change: Reduce batch size from 64 to 48, interpolate the learning rate to 2.25e-3, and replace the broader ±2-pixel crop grid with the five views used at inference.
mechanism: Runtime-safe intermediate-batch matched-view optimization
evidence_used: Reference Design 3 reached 9,214 correct using batch-32 dense optimization and matched cardinal views, while the current batch-64 design finishes in 63.63 seconds; batch 48 captures additional updates and view matching without the timeout-prone 3,126-step regime.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch-64 paired-flip training with matched cardinal crops and ten-view inference will exceed 9,119 correct predictions while remaining safely faster than the timeout-prone batch-32 designs.
change: Double optimizer updates, supervise both horizontal orientations, align training crops with inference views, add pair-batched ten-view inference, and use the proven batch-64 learning rate.
mechanism: Runtime-funded paired-view densification
evidence_used: The current model reached 9,105 correct in only 22.3 seconds, leaving substantial runtime headroom; Reference Design 2 safely completed batch-64 ten-view evaluation with 9,119 correct, while Reference Design 3 showed matched cardinal crops and paired flips can reach 9,214.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 58.7689845408313, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.24051190643310547, "validation_score": 9232.403059412334}

RECENT RESULT
hypothesis: Adding a mild, zero-forward-cost consistency penalty between paired original and mirrored predictions to Reference Design 1 will exceed 9,232 correct predictions without materially increasing its 58.8-second runtime.
change: Adopt the strongest verified batch-64 architecture, matched cardinal crops, paired-flip supervision, cosine schedule, and ten-view inference, then regularize each flip pair toward a shared predictive distribution.
mechanism: Flip-orbit Jensen–Shannon regularization
evidence_used: Reference Design 1 achieved the best result at 9,232 correct with paired flips and mirrored test-time averaging; directly reducing disagreement between those paired predictions targets the same symmetry while retaining its demonstrated runtime headroom.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Weighting spatial inference views according to their 60/10/10/10/10 training exposure will exceed 9,232 correct predictions by retaining shift robustness while reducing the influence of rarely trained shifted views.
change: Replace uniform ten-view probability averaging with a weighted mixture: 60% total weight for the centered view and 10% for each cardinal shift, split equally between original and mirrored orientations.
mechanism: Exposure-matched test-time view weighting
evidence_used: The current design achieved 9,232 correct, while shifted ten-view inference improved Reference Design 3 by only seven correct. Because training exposes centered images 60% of the time but uniform inference gives them only 20% weight, matching inference weights to the observed training distribution is a compute-neutral refinement of that marginal TTA benefit.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the proven 247,538-parameter batch-64 design will exceed the current design’s 9,214 correct predictions while completing within the verification limit.
change: Replace the wider residual network with the verified 24/48/96-channel model, use batch 64 and its 2.5e-3 schedule, and batch each original/mirrored inference pair into one forward pass.
mechanism: Verified batch-64 width-and-update allocation
evidence_used: Reference Design 2 achieved the best available result—9,232 correct with 0.2405 cross-entropy—in 58.8 seconds; the current batch-32 residual design achieved only 9,214 correct and took 81.3 seconds.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Restoring the exact 247,538-parameter batch-64 training design will exceed the current 9,119 correct predictions and reproduce approximately 9,232 correct within the demonstrated runtime.
change: Replace the wider residual network with the verified 24/48/96-channel model, align early training crops with the five inference views, and supervise original/mirrored image pairs jointly.
mechanism: Verified width-and-update allocation with paired symmetry supervision
evidence_used: Reference Design 2 achieved the best verified result—9,232 correct and 0.2405 cross-entropy—in 58.8 seconds, outperforming the current 9,119-correct design while satisfying the runtime and parameter constraints.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 47.30621508299373, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.24051190643310547, "validation_score": 9232.403059412334}

RECENT RESULT
hypothesis: The verified batch-64 design with a 2% terminal learning-rate floor will complete within its demonstrated runtime headroom and exceed 9,232 correct predictions by reducing late-training parameter noise.
change: Restore the 247,538-parameter paired-flip, cardinal-crop, ten-view design and lower its cosine schedule floor from 10% to 2%.
mechanism: Low-floor cosine stabilization
evidence_used: Reference Designs 1 and 2 reached 9,232 correct in 47.3–58.8 seconds; a 0.995 EMA reached 9,241 before timing out, motivating a zero-overhead stabilization change on the runtime-safe batch-64 design.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 56.48495883401483, "validation_accuracy": 0.9243, "validation_correct": 9243, "validation_cross_entropy": 0.23800013732910155, "validation_score": 9243.403877176523}

RECENT RESULT
hypothesis: Extending the successful learning-rate-floor reduction from 2% to zero will exceed 9,243 correct predictions by further suppressing late-training parameter noise without increasing runtime.
change: Replace the 2% terminal learning-rate floor with a pure cosine decay to zero.
mechanism: Zero-floor cosine settling
evidence_used: Lowering the floor from 10% to 2% improved validation correctness from 9,232 to 9,243 at the same batch size, architecture, and example budget, supporting a compute-neutral test of the zero-floor boundary.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 76.11452154186554, "validation_accuracy": 0.9209, "validation_correct": 9209, "validation_cross_entropy": 0.24160904960632323, "validation_score": 9209.402703250398}



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
