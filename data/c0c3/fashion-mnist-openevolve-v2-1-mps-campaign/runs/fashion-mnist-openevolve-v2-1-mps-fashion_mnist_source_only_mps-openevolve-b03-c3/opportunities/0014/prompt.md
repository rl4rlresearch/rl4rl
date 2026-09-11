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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 36.624056624947116, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.2430596939086914, "validation_score": 9229.402233297766}
prior_hypothesis: Blending per-view cross-entropy with cross-entropy on each flip pair’s averaged logits will exceed 9,224 correct predictions by directly optimizing the flip-averaged decision rule used during validation while retaining robust individual views.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241682, "training_seconds": 30.0022278749384, "validation_accuracy": 0.9186, "validation_correct": 9186, "validation_cross_entropy": 0.2547276756286621, "validation_score": 9186.398492844075}
prior_hypothesis: Restoring the 9,155-correct weighted-translation/logit-ensemble design and supervising both horizontal orientations of every sampled crop will exceed 9,155 correct predictions by eliminating flip-sampling variance and aligning training with the proven flip-averaged evaluation.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 32.57326712505892, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.2523522033691406, "validation_score": 9191.399248708674}
prior_hypothesis: Applying a 0.99-decay EMA to the 9,229-correct differential-basis, flip-ensemble-supervised model will exceed 9,229 correct predictions by reducing sensitivity to the final noisy AdamW iterate.

## Recent verification evidence

RECENT RESULT
hypothesis: Reallocating the reference model’s large dense layer into residual spatial convolutions, while increasing optimizer updates with batch size 128, will exceed 9,009 correct validation predictions within the same 100,000-example budget.
change: Build a 247,794-parameter three-stage residual CNN, retain proven translation/flip augmentation and flip-averaged evaluation, and train it with decoupled AdamW, mild label smoothing, and cosine decay.
mechanism: Convolution-heavy residual hierarchy with spatially pooled classification
evidence_used: The 241,682-parameter reference reached 9,009 correct, but 188,160 weights were concentrated in its first dense layer; using that capacity for hierarchical convolution should learn stronger shape features while preserving its validated augmentation and evaluation strategy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding weighted ±1-pixel translation averaging to the proven flip ensemble will exceed 9,009 correct predictions by reducing sensitivity to alignment, without changing the validated 241,682-parameter training recipe.
change: Adopt Reference Design 1 and extend evaluation to average centered and four one-pixel-shifted views, each paired with its horizontal flip.
mechanism: Translation-orbit test-time ensembling
evidence_used: Reference Design 1 achieved 9,009 correct using random translations during training and flip-averaged evaluation; translation-aware evaluation directly exploits the invariance already learned by that verified model.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241682, "training_seconds": 23.97421450004913, "validation_accuracy": 0.9033, "validation_correct": 9033, "validation_cross_entropy": 0.2854304672241211, "validation_score": 9033.388974754176}

RECENT RESULT
hypothesis: Extending the verified center/cardinal flip ensemble to include lightly weighted diagonal one-pixel translations will exceed 9,033 correct predictions by covering alignment variations omitted from Reference Design 2.
change: Adopt Reference Design 2’s 241,682-parameter training recipe and evaluate a weighted 3×3 translation neighborhood paired with horizontal flips.
mechanism: Diagonal-aware translation-orbit test-time ensembling
evidence_used: Reference Design 2 improved from 9,009 to 9,033 correct solely by adding cardinal translation averaging, while training already exposes the model to independently sampled horizontal and vertical shifts, including diagonal translations.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241682, "training_seconds": 24.704568667104468, "validation_accuracy": 0.9042, "validation_correct": 9042, "validation_cross_entropy": 0.28424416427612303, "validation_score": 9042.389334064275}

RECENT RESULT
hypothesis: Adding a second lightweight residual refinement block will exceed 9,042 correct predictions by using the remaining parameter budget to deepen spatial feature processing while preserving the verified training and translation-ensemble recipe.
change: Add a second 80-channel depthwise-separable residual block and apply the two refinement blocks sequentially, increasing learned parameters from 241,682 to 249,122.
mechanism: Stacked depthwise-separable residual refinement
evidence_used: The current 241,682-parameter model achieves 9,042 correct and leaves 8,318 parameters unused; an identical refinement block costs only 7,440 parameters, isolating the value of additional convolutional depth without disturbing the proven optimizer, augmentation, or evaluation ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249122, "training_seconds": 36.02561662485823, "validation_accuracy": 0.9039, "validation_correct": 9039, "validation_cross_entropy": 0.2833393844604492, "validation_score": 9039.389608552543}

RECENT RESULT
hypothesis: Extending the 9,042-correct 3×3 ensemble with lightly weighted two-pixel translations will exceed 9,042 correct predictions because training already samples the full ±2-pixel translation range.
change: Evaluate horizontal-flip pairs across the complete 5×5 translation neighborhood, retaining most weight on the proven center and one-pixel views while assigning 10% total weight to the radius-two ring.
mechanism: Lightly weighted radius-two translation-orbit ensembling
evidence_used: Reference Design 3 improved from 9,033 to 9,042 correct by adding previously omitted diagonal translations; this motivates testing the remaining translations represented by the training augmentation without changing the proven model or optimizer.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241682, "training_seconds": 21.776120292022824, "validation_accuracy": 0.9051, "validation_correct": 9051, "validation_cross_entropy": 0.28426247100830077, "validation_score": 9051.38932851445}

RECENT RESULT
hypothesis: Replacing logit averaging with a weighted probability mixture over the proven 5×5 translation-and-flip ensemble will exceed 9,051 correct predictions by preventing weak radius-two views from disproportionately suppressing confident class probabilities.
change: Add the best verified 5×5 translation ensemble and combine its flip-paired predictions as probabilities, returning normalized log-probabilities for validation.
mechanism: Probability-space translation-orbit mixture
evidence_used: The weighted 5×5 logit ensemble reached 9,051 correct, improving on both the 9,042-correct 3×3 ensemble and the 9,009-correct flip-only model; its slightly higher cross-entropy than the 3×3 result suggests aggregation, rather than view coverage, is the next useful variable.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241682, "training_seconds": 20.509318750118837, "validation_accuracy": 0.9047, "validation_correct": 9047, "validation_cross_entropy": 0.28850603866577146, "validation_score": 9047.388046299355}

RECENT RESULT
hypothesis: Sampling training translations with the proven 5×5 evaluation weights will exceed 9,051 correct predictions by retaining radius-two invariance while focusing exposure on the more reliable center and radius-one views.
change: Replace uniform 5×5 translation sampling with the exact center/cardinal/diagonal/radius-two distribution used by the best verified logit ensemble.
mechanism: Reliability-weighted translation augmentation
evidence_used: The weighted 5×5 logit ensemble improved from 9,042 to 9,051 correct, showing radius-two views help when lightly weighted; uniform training currently assigns them 64% of examples despite their 10% evaluation weight.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241682, "training_seconds": 28.43675791588612, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.26254745712280275, "validation_score": 9155.396024717471}

RECENT RESULT
hypothesis: Restoring the 9,155-correct weighted-translation/logit-ensemble design and supervising both horizontal orientations of every sampled crop will exceed 9,155 correct predictions by eliminating flip-sampling variance and aligning training with the proven flip-averaged evaluation.
change: Use reliability-weighted translation sampling, restore weighted logit averaging at evaluation, and train each sampled crop together with its horizontal flip in one forward pass.
mechanism: Flip-orbit-complete supervised minibatches
evidence_used: Reference Design 2 achieved 9,155 correct with weighted translation sampling and logit aggregation, while probability aggregation reduced performance to 9,047; flip averaging is already proven at evaluation, motivating complete paired flip exposure during training.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241682, "training_seconds": 30.0022278749384, "validation_accuracy": 0.9186, "validation_correct": 9186, "validation_cross_entropy": 0.2547276756286621, "validation_score": 9186.398492844075}

RECENT RESULT
hypothesis: Replacing the shared raw-pixel-only front end with raw, local-contrast, signed-gradient, and edge-energy channels—while retaining the 9,186-correct training and evaluation recipe—will exceed 9,186 correct predictions by exposing shape evidence directly during the limited 522 optimizer steps.
change: Challenge the assumption that another raw-pixel CNN refinement is the best use of capacity: learn over a fixed five-channel differential image basis, restore reliability-weighted translation sampling and 5×5 logit aggregation, and supervise both flip orientations.
mechanism: Differential-basis contour encoding
evidence_used: Flip-complete weighted-translation training reached 9,186 correct, whereas adding another same-type residual refinement reached only 9,039; this favors retaining the proven invariance recipe while testing a genuinely different input representation rather than adding more conventional depth.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 38.74637974984944, "validation_accuracy": 0.9224, "validation_correct": 9224, "validation_cross_entropy": 0.24455272064208985, "validation_score": 9224.401750758892}

RECENT RESULT
hypothesis: Blending per-view cross-entropy with cross-entropy on each flip pair’s averaged logits will exceed 9,224 correct predictions by directly optimizing the flip-averaged decision rule used during validation while retaining robust individual views.
change: Replace purely independent flip-pair supervision with an equal mixture of individual-view and flip-averaged logit losses.
mechanism: Flip-orbit ensemble supervision
evidence_used: Complete flip-pair training improved validation from 9,155 to 9,186 correct, and the current model evaluates every translation through flip-averaged logits; directly supervising that same aggregation is the next isolated test of objective alignment.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 36.624056624947116, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.2430596939086914, "validation_score": 9229.402233297766}

RECENT RESULT
hypothesis: Extending the 9,229-correct differential-basis model from flip-pair supervision to symmetric translation-pair supervision will exceed 9,229 correct predictions by directly optimizing a four-view approximation of the weighted translation-and-flip ensemble used for validation.
change: Restore the verified five-channel differential image basis, pair every sampled translation with its inverse offset, and jointly supervise individual views, flip-pair averages, and the complete four-view translation-orbit average.
mechanism: Antithetic translation-orbit ensemble supervision
evidence_used: Reference Design 3 reached 9,229 correct after flip-averaged loss improved the 9,224-correct differential-basis model; earlier weighted translation sampling and weighted logit evaluation also produced large gains, motivating the same objective-alignment strategy across symmetric translations.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Applying a 0.99-decay EMA to the 9,229-correct differential-basis, flip-ensemble-supervised model will exceed 9,229 correct predictions by reducing sensitivity to the final noisy AdamW iterate.
change: Restore the strongest differential-image-basis and flip-ensemble loss, then maintain an EMA of learned parameters and floating-point BatchNorm buffers and install it after the final optimizer step.
mechanism: Exponential trajectory weight averaging
evidence_used: Reference Design 3 reached 9,229 correct after aligning training with flip-averaged evaluation; its cosine schedule retains a nonzero learning-rate floor, motivating trajectory averaging as an isolated way to stabilize the final classifier without adding learned parameters.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 32.57326712505892, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.2523522033691406, "validation_score": 9191.399248708674}



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
