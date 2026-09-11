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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 49.27608129195869, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.23900571517944336, "validation_score": 9235.403549389543}
prior_hypothesis: A 3% terminal learning-rate floor will retain the late-training stability of the best 2% floor while avoiding slight over-settling, exceeding 9,243 correct predictions.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 73.43461545812897, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.23899341583251954, "validation_score": 9249.403553395532}
prior_hypothesis: A 2.2% terminal learning-rate floor will exceed 9,243 correct predictions by approaching the local optimum implied by the 1.8%, 2%, and 3% results.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 52.015812791883945, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.23853298721313476, "validation_score": 9242.403703417804}
prior_hypothesis: A 2.15% terminal learning-rate floor will exceed 9,249 correct predictions by targeting the quadratic peak implied by the verified 2.0%, 2.2%, and 2.4% results.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 73.53027954208665, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.23828823318481446, "validation_score": 9241.403783211857}
prior_hypothesis: A 1.8% terminal learning-rate floor will exceed 9,243 correct predictions by preserving the stability of the best 2% schedule while testing the empirical optimum implied by the 1%, 2%, and 3% results.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 1% terminal learning-rate floor will retain the stability lost at zero while further reducing late-update noise, exceeding 9,243 correct predictions.
change: Change the cosine schedule’s terminal multiplier from 10% to 1%, leaving the verified architecture, optimizer, augmentation, and inference unchanged.
mechanism: Nonzero-floor cosine boundary refinement
evidence_used: The 2% floor achieved the best result at 9,243 correct, while zero fell to 9,209; testing the midpoint isolates whether a smaller strictly positive floor improves the established optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 68.27344562509097, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.23947361679077148, "validation_score": 9239.403397049544}

RECENT RESULT
hypothesis: A 3% terminal learning-rate floor will retain the late-training stability of the best 2% floor while avoiding slight over-settling, exceeding 9,243 correct predictions.
change: Change the cosine schedule’s terminal multiplier from 10% to 3%, leaving architecture, optimizer, augmentation, and inference unchanged.
mechanism: Nonzero-floor cosine upper-bound refinement
evidence_used: The 2% floor achieved 9,243 correct, outperforming both the 1% floor’s 9,239 and zero floor’s 9,209; testing 3% probes the untested upper neighbor of the established optimum with no runtime cost.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 49.27608129195869, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.23900571517944336, "validation_score": 9235.403549389543}

RECENT RESULT
hypothesis: A 1.8% terminal learning-rate floor will exceed 9,243 correct predictions by preserving the stability of the best 2% schedule while testing the empirical optimum implied by the 1%, 2%, and 3% results.
change: Restore the verified 247,538-parameter batch-64 paired-flip design and use a cosine schedule ending at 1.8% of the initial learning rate.
mechanism: Fine-grained cosine-floor interpolation
evidence_used: The 2% floor achieved 9,243 correct versus 9,239 at 1% and 9,235 at 3%; a local quadratic interpolation of those correctness results peaks near 1.83%, motivating a compute-neutral 1.8% test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 73.53027954208665, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.23828823318481446, "validation_score": 9241.403783211857}

RECENT RESULT
hypothesis: A 2.2% terminal learning-rate floor will exceed 9,243 correct predictions by approaching the local optimum implied by the 1.8%, 2%, and 3% results.
change: Raise the verified cosine schedule’s terminal multiplier from 2.0% to 2.2%, leaving all other training and inference behavior unchanged.
mechanism: Fine-grained cosine-floor peak refinement
evidence_used: The 2% floor achieved the best result at 9,243 correct, versus 9,241 at 1.8% and 9,235 at 3%; local interpolation of these neighboring results places the estimated peak near 2.2%.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 73.43461545812897, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.23899341583251954, "validation_score": 9249.403553395532}

RECENT RESULT
hypothesis: A 2.4% terminal learning-rate floor will exceed 9,249 correct predictions by approaching the quadratic optimum implied by the 2.0%, 2.2%, and 3.0% results.
change: Raise the cosine schedule’s terminal multiplier from 1.8% to 2.4%, leaving architecture, optimizer, augmentation, and inference unchanged.
mechanism: Cosine-floor peak interpolation
evidence_used: The 2.2% floor achieved the best result at 9,249 correct, versus 9,243 at 2.0% and 9,235 at 3.0%; quadratic interpolation places the estimated peak near 2.42%.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 77.75563320796937, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.23861759567260743, "validation_score": 9232.403675841315}

RECENT RESULT
hypothesis: A 2.15% terminal learning-rate floor will exceed 9,249 correct predictions by targeting the quadratic peak implied by the verified 2.0%, 2.2%, and 2.4% results.
change: Change the cosine schedule’s terminal multiplier from 1.0% to 2.15%, leaving architecture, optimizer, augmentation, and inference unchanged.
mechanism: Cosine-floor vertex refinement
evidence_used: The 2.2% floor achieved 9,249 correct, versus 9,243 at 2.0% and 9,232 at 2.4%; quadratic interpolation of these three observations places the estimated peak near 2.15%.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 52.015812791883945, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.23853298721313476, "validation_score": 9242.403703417804}



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
