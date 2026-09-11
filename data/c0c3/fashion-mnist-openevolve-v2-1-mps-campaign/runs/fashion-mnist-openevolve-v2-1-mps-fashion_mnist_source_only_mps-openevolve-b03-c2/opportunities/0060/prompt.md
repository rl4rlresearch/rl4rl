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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 74.72353874985129, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.23852134895324706, "validation_score": 9241.403707211364}
prior_hypothesis: A 2.25% terminal learning-rate floor will exceed 9,249 correct predictions by targeting the quadratic vertex implied by the verified 2.15%, 2.2%, and 2.4% results.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 73.43461545812897, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.23899341583251954, "validation_score": 9249.403553395532}
prior_hypothesis: A 2.2% terminal learning-rate floor will exceed 9,243 correct predictions by approaching the local optimum implied by the 1.8%, 2%, and 3% results.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.29047845909372, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.2366387908935547, "validation_score": 9249.404321782304}
prior_hypothesis: Restoring the best 2.2% cosine floor and giving the center view twice each shifted view’s inference weight will exceed 9,249 correct predictions by reducing influence from less frequently trained translations.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 57.225683541968465, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.23508031921386718, "validation_score": 9253.404831971024}
prior_hypothesis: A 3:1 center-to-shift inference weight with the verified 2.2% cosine floor will exceed 9,249 correct predictions by further reducing the influence of translations that appear less often during training.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 2.25% terminal learning-rate floor will exceed 9,249 correct predictions by targeting the quadratic vertex implied by the verified 2.15%, 2.2%, and 2.4% results.
change: Lower the current cosine schedule floor from 3% to 2.25%, leaving architecture, optimizer, augmentation, and inference unchanged.
mechanism: Updated cosine-floor vertex refinement
evidence_used: The 2.2% floor achieved the best result at 9,249 correct, versus 9,242 at 2.15% and 9,232 at 2.4%; quadratic interpolation of these closest observations places the estimated peak near 2.25%.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 74.72353874985129, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.23852134895324706, "validation_score": 9241.403707211364}

RECENT RESULT
hypothesis: A 2.198% terminal learning-rate floor will exceed 9,249 correct predictions by targeting the quadratic vertex defined by the verified 2.15%, 2.20%, and 2.25% results.
change: Lower the cosine schedule’s terminal multiplier from 2.2% to 2.198%, leaving all other training and inference behavior unchanged.
mechanism: Local cosine-floor vertex refinement
evidence_used: The 2.2% floor achieved 9,249 correct, while the symmetric neighboring 2.15% and 2.25% floors achieved 9,242 and 9,241; quadratic interpolation places the local vertex at approximately 2.198%.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.28309900010936, "validation_accuracy": 0.9241, "validation_correct": 9241, "validation_cross_entropy": 0.23883129844665527, "validation_score": 9241.403606205806}

RECENT RESULT
hypothesis: Restoring the best 2.2% cosine floor and giving the center view twice each shifted view’s inference weight will exceed 9,249 correct predictions by reducing influence from less frequently trained translations.
change: Use the verified 2.2% terminal learning-rate floor and change ten-view probability averaging to a normalized 2:1 center-to-shift weighting without additional forward passes.
mechanism: Center-emphasized test-time augmentation
evidence_used: The 2.2% floor achieved the best result at 9,249 correct, while nearby floor refinements all regressed. The center crop supplies 60% of training batches but currently receives only 20% of aggregate inference weight, motivating a new inference-weighting test while retaining the strongest schedule.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 61.29047845909372, "validation_accuracy": 0.9249, "validation_correct": 9249, "validation_cross_entropy": 0.2366387908935547, "validation_score": 9249.404321782304}

RECENT RESULT
hypothesis: A 3:1 center-to-shift inference weight with the verified 2.2% cosine floor will exceed 9,249 correct predictions by further reducing the influence of translations that appear less often during training.
change: Restore the best 2.2% learning-rate floor and increase the center view’s inference weight from the verified 2:1 setting to 3:1 without adding forward passes.
mechanism: Stronger center-emphasized test-time augmentation
evidence_used: The 2:1 center-weighted design tied the best 9,249 correct result while lowering cross-entropy from 0.2390 to 0.2366 versus uniform averaging, establishing center emphasis as beneficial and motivating a directional weight increase.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247538, "training_seconds": 57.225683541968465, "validation_accuracy": 0.9253, "validation_correct": 9253, "validation_cross_entropy": 0.23508031921386718, "validation_score": 9253.404831971024}



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
