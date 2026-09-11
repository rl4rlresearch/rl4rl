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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 61.61174912494607, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.2466726921081543, "validation_score": 9133.40106758026}
prior_hypothesis: Blending 28.125% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,130 correct predictions by moving slightly beyond the best observed 25% blend while remaining well below the degraded 50% blend.

## Recent verification evidence

RECENT RESULT
hypothesis: Sampling an independent ±2-pixel translation for every training image will exceed 9,073 correct predictions by reducing correlated augmentation noise within each batch.
change: Replace the single batch-wide crop offset with vectorized per-image crop selection while preserving the augmentation range, batch size, architecture, and optimizer.
mechanism: Per-example translation diversification
evidence_used: Ten-view translation-and-flip inference improved the current design from 9,050 to 9,073 correct, showing translation variance remains consequential; training currently gives all 64 images the same translation, so independent offsets directly target that variance.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 44.937164624920115, "validation_accuracy": 0.9082, "validation_correct": 9082, "validation_cross_entropy": 0.25911584167480467, "validation_score": 9082.397104049882}

RECENT RESULT
hypothesis: Averaging all 50 views from the training augmentation distribution will exceed 9,082 correct predictions by further reducing translation-sensitive errors.
change: Replace the ten-view cross-shaped ensemble with every combination of ±2-pixel horizontal and vertical translation, each with its horizontal flip.
mechanism: Full-support translation-and-flip test-time ensembling
evidence_used: Ten-view translation-and-flip inference improved accuracy from 9,050 to 9,073, and per-image sampling across the full ±2 training range then improved it to 9,082; evaluating the complete matched transformation distribution directly extends both gains.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 53.59044679207727, "validation_accuracy": 0.9074, "validation_correct": 9074, "validation_cross_entropy": 0.26268745346069333, "validation_score": 9074.395980809526}

RECENT RESULT
hypothesis: Biasing training crops toward zero- and one-pixel translations while retaining occasional two-pixel shifts will exceed 9,082 correct predictions.
change: Replace uniform ±2-pixel offsets with independent triangular offsets formed by summing two uniform three-way samples.
mechanism: Center-weighted per-image translation augmentation
evidence_used: Ten-view inference using central and one-pixel translations reached 9,082 correct, whereas uniformly averaging the full ±2 support fell to 9,074, indicating that extreme transformations should receive less weight.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 59.7259305419866, "validation_accuracy": 0.9088, "validation_correct": 9088, "validation_cross_entropy": 0.25145248794555664, "validation_score": 9088.39953574332}

RECENT RESULT
hypothesis: Weighting the centered evaluation view 3:2 relative to each one-pixel cardinal translation will exceed 9,088 correct predictions by matching the successful triangular training-crop distribution.
change: Change the ten-view test-time ensemble from equal averaging to triangular-distribution weights while retaining the same views and horizontal flips.
mechanism: Center-weighted matched test-time ensembling
evidence_used: Center-weighted training translations improved accuracy from 9,082 to 9,088, while uniformly averaging the full translation support fell to 9,074; this motivates testing center bias at inference without changing training.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 61.90971816610545, "validation_accuracy": 0.9091, "validation_correct": 9091, "validation_cross_entropy": 0.2513663063049316, "validation_score": 9091.39956325936}

RECENT RESULT
hypothesis: Exposing raw intensity, horizontal and vertical Sobel responses, and Laplacian detail will exceed 9,091 correct predictions because the fixed exposure budget no longer requires the shallow stem to learn these shape-sensitive primitives from pixels alone.
change: Challenge the raw-pixel-only representation assumption by adding three parameter-free differential channels before the existing CNN and expanding its first convolution from one to four inputs, for approximately 245,034 learned parameters.
mechanism: Fixed differential-basis image representation
evidence_used: Doubling optimizer updates left accuracy tied at 9,050 while increasingly refined spatial ensembling produced only incremental gains to 9,091, suggesting optimization and translation variance are no longer the main bottlenecks; a fixed edge/detail basis cleanly tests whether more sample-efficient image representation can provide the next gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 38.54385858308524, "validation_accuracy": 0.9122, "validation_correct": 9122, "validation_cross_entropy": 0.2498493896484375, "validation_score": 9122.40004820112}

RECENT RESULT
hypothesis: Evaluating a short-horizon EMA of the noisy batch-64 training trajectory will exceed 9,122 correct predictions while lowering validation cross-entropy.
change: Track an adaptive 0.99-decay EMA after every optimizer update and copy the averaged weights into the model after the final fixed-budget step.
mechanism: Bias-corrected exponential moving average of learned weights
evidence_used: Per-example stochastic translations raised accuracy from 9,073 to 9,082, and differential features raised it further to 9,122; averaging the resulting stochastic weight trajectory is an orthogonal way to retain these gains while reducing optimizer and augmentation variance.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 45.71527612488717, "validation_accuracy": 0.9121, "validation_correct": 9121, "validation_cross_entropy": 0.2454588264465332, "validation_score": 9121.401458474083}

RECENT RESULT
hypothesis: Adding parameter-free 5×5 blur, gradient, and center-surround channels will exceed 9,122 correct predictions by extending the successful fixed representation with broader shape context while remaining below the parameter ceiling.
change: Expand the input representation from four to eight channels using four fixed 5×5 filters and widen only the first convolution, increasing learned parameters from 245,034 to 245,898.
mechanism: Multiscale fixed differential-basis representation
evidence_used: Adding 3×3 Sobel and Laplacian channels produced the largest recent gain, from 9,091 to 9,122 correct, indicating that fixed shape-sensitive preprocessing improves sample efficiency; broader filters directly test whether additional spatial scale yields another gain.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245898, "training_seconds": 49.815060207853094, "validation_accuracy": 0.9113, "validation_correct": 9113, "validation_cross_entropy": 0.24972968521118163, "validation_score": 9113.400086519443}

RECENT RESULT
hypothesis: Blending 25% of a bias-corrected 0.99 EMA into the final weights will exceed the current 9122.400048 validation score by preserving endpoint accuracy while inheriting some of the EMA’s lower cross-entropy.
change: Track a parameter EMA after every optimizer step, then interpolate the final parameters 25% toward that EMA only at the last fixed-budget step.
mechanism: Endpoint-anchored short-horizon weight averaging
evidence_used: Full EMA produced 9,121 correct versus the endpoint’s 9,122, but improved cross-entropy substantially from 0.24985 to 0.24546; this indicates useful trajectory smoothing whose full-strength replacement was slightly too aggressive.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 49.29732579202391, "validation_accuracy": 0.913, "validation_correct": 9130, "validation_cross_entropy": 0.2468909523010254, "validation_score": 9130.400997375975}

RECENT RESULT
hypothesis: Increasing the EMA interpolation from 25% to 50% will exceed 9,130 correct predictions by capturing more trajectory smoothing without the accuracy loss caused by full EMA replacement.
change: Blend final parameters halfway toward the bias-corrected 0.99 EMA instead of one-quarter toward it.
mechanism: Midpoint endpoint–EMA weight interpolation
evidence_used: The endpoint achieved 9,122 correct, 25% EMA blending improved this to 9,130, and full EMA achieved 9,121 with substantially lower cross-entropy; these results indicate an interior blend is best and motivate testing the midpoint.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 55.52620549988933, "validation_accuracy": 0.9123, "validation_correct": 9123, "validation_cross_entropy": 0.2457532669067383, "validation_score": 9123.401363587223}

RECENT RESULT
hypothesis: Blending 18.75% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,130 correct predictions by retaining more of the endpoint than the successful 25% blend while preserving useful trajectory smoothing.
change: Reduce the final EMA interpolation coefficient from 0.25 to 0.1875 without changing training, architecture, or evaluation.
mechanism: Lower-side endpoint–EMA interpolation refinement
evidence_used: The 25% blend improved accuracy from 9,122 to 9,130, whereas 50% and 100% blends fell to 9,123 and 9,121; this brackets the best observed region at or below 25% and motivates a lower-side refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 69.93643308407627, "validation_accuracy": 0.9129, "validation_correct": 9129, "validation_cross_entropy": 0.24741411895751952, "validation_score": 9129.400829197299}

RECENT RESULT
hypothesis: Blending 21.875% of the bias-corrected EMA into the endpoint will exceed 9,130 correct predictions by refining the narrow optimum bracketed by the 18.75% and 25% results.
change: Reduce the final EMA interpolation coefficient from 0.25 to 0.21875 while leaving training and evaluation unchanged.
mechanism: Fine-grained endpoint–EMA interpolation
evidence_used: The 25% blend achieved 9,130 correct, while 18.75% achieved 9,129 and 50% fell to 9,123, indicating a narrow optimum near 25% and motivating a midpoint refinement on its better-supported lower side.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 71.84408129099756, "validation_accuracy": 0.913, "validation_correct": 9130, "validation_cross_entropy": 0.24713736572265624, "validation_score": 9130.40091814562}

RECENT RESULT
hypothesis: Blending 28.125% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,130 correct predictions by moving slightly beyond the best observed 25% blend while remaining well below the degraded 50% blend.
change: Increase the final EMA interpolation coefficient from 0.25 to 0.28125 without changing training, architecture, or evaluation.
mechanism: Upper-side endpoint–EMA interpolation refinement
evidence_used: Accuracy increased from 9,129 at 18.75% blending to 9,130 at both 21.875% and 25%, with 25% also giving better cross-entropy; 50% fell to 9,123, so 28.125% is the nearest untested upper-side refinement around the apparent optimum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 61.61174912494607, "validation_accuracy": 0.9133, "validation_correct": 9133, "validation_cross_entropy": 0.2466726921081543, "validation_score": 9133.40106758026}



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
