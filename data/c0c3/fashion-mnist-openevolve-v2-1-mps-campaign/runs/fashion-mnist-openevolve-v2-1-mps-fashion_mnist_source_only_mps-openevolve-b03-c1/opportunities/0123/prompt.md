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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 67.21863225009292, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22713947525024414, "validation_score": 9206.407451646764}
prior_hypothesis: Eliminating diagonal training translations while preserving balanced one-pixel cardinal shifts will exceed 9,196 correct predictions by aligning augmentation support with the successful validation views.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing diagonal training shifts while retaining cardinal shifts up to two pixels will exceed 9,166 correct predictions by aligning augmentation with the successful cardinal validation views without sacrificing broad translation invariance.
change: Sample each training translation along exactly one randomly chosen axis, preserving the existing triangular ±2 displacement distribution, flips, optimizer, model, and runtime profile.
mechanism: Axis-only broad translation augmentation
evidence_used: Adding diagonal validation views reduced correctness from 9,166 to 9,155. The timed-out curriculum also changed phase and shift magnitude; this patch isolates diagonal-support removal while retaining the baseline’s broad augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Training `detail_kernels` at 0.70× the base learning rate will exceed 9,166 correct predictions by balancing useful task-specific adaptation against the degradation observed at 2× learning rate.
change: Place `detail_kernels` in a separate AdamW parameter group with a persistent 0.70× learning-rate multiplier, leaving weight decay, scheduling, EMA, architecture, and runtime otherwise unchanged.
mechanism: Moderated detail-filter adaptation rate
evidence_used: Learning the kernels improved correctness from 9,162 to 9,166, but doubling their learning rate reduced it to 9,149; an intermediate slower rate tests whether the current kernels over-adapt slightly while retaining the benefit lost when they are fixed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Increasing the centered validation-view weight from 3.0 to 3.25 will exceed 9,166 correct predictions by exploiting the asymmetric results around the current setting while avoiding the excessive center dominance of weight 4.0.
change: Change only the centered view’s ensemble weight from 3.0 to 3.25, preserving training, model parameters, calibration, and runtime.
mechanism: Fine-grained center-weighted logit ensemble
evidence_used: Center weights 2.5 and 4.0 produced 9,163 and 9,164 correct respectively, versus 9,166 at 3.0; the smaller regression above 3.0 motivates a conservative upward refinement within the tested interval.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 62.922055166913196, "validation_accuracy": 0.9164, "validation_correct": 9164, "validation_cross_entropy": 0.23898156661987305, "validation_score": 9164.403557254984}

RECENT RESULT
hypothesis: Adding a lightweight residual spatial gate to the final 3×3 feature map will exceed 9,166 correct predictions by learning which spatial cells contain discriminative evidence while preserving the successful flattened classifier at initialization.
change: Add a 19-parameter mean–maximum spatial attention convolution after channel gating, zero-initialized to an identity transformation, raising the model to 249,808 parameters.
mechanism: Zero-initialized spatial attention gate
evidence_used: Global-context classification regressed to 9,128, showing that spatially resolved classification is important; this patch retains that representation and adds only adaptive spatial reweighting, unlike the heavier refinement and multiscale branches that could not be verified.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249808, "training_seconds": 74.35137004218996, "validation_accuracy": 0.9134, "validation_correct": 9134, "validation_cross_entropy": 0.2421733486175537, "validation_score": 9134.402520308906}

RECENT RESULT
hypothesis: Restricting training translations from ±2 to ±1 pixels will exceed 9,166 correct predictions by reducing transform mismatch and preserving more discriminative edge detail during the fixed two-pass exposure.
change: Change both training crop offsets to a triangular −1/0/+1 distribution while retaining diagonal shifts, flips, and all model and optimization settings.
mechanism: Triangular one-pixel translation augmentation
evidence_used: The successful validation ensemble uses only ±1 cardinal shifts, while adding diagonal views reduced correctness from 9,166 to 9,155; isolating shift magnitude tests whether the broader ±2 training support is excessive without confounding architecture or runtime.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 77.02114270906895, "validation_accuracy": 0.9196, "validation_correct": 9196, "validation_cross_entropy": 0.22880392379760742, "validation_score": 9196.406899742356}

RECENT RESULT
hypothesis: Replacing diagonal training shifts with center/cardinal ±1 shifts weighted like the successful validation ensemble will exceed 9,196 correct predictions by retaining beneficial translation invariance while eliminating unsupported diagonal transforms.
change: Sample only the centered crop and four one-pixel cardinal translations, with probabilities proportional to validation weights 3:2:2:2:2; retain flips and all other training settings.
mechanism: Center-and-cardinal translation augmentation
evidence_used: Restricting translations from ±2 to ±1 improved correctness from 9,166 to 9,196, while adding diagonal validation views previously reduced correctness to 9,155; this directly motivates removing the remaining diagonal training transforms.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reassigning diagonal one-pixel training shifts to the centered crop will exceed 9,196 correct predictions by eliminating transforms unsupported by the successful validation ensemble while preserving the baseline frequency of each cardinal shift.
change: Keep the existing triangular ±1 sampling, but convert samples shifted on both axes to centered crops, changing the distribution from 25% center/50% cardinal/25% diagonal to 50% center/50% cardinal/0% diagonal.
mechanism: Diagonal-to-center translation redistribution
evidence_used: Restricting translations from ±2 to ±1 improved correctness from 9,166 to 9,196, while diagonal validation views previously reduced correctness to 9,155. Unlike the timed-out 3:2:2:2:2 proposal, this isolates diagonal removal without increasing total cardinal-shift exposure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reassigning diagonal one-pixel shifts to cardinal shifts will exceed 9,196 correct predictions by eliminating unsupported diagonal transforms while closely matching the successful validation ensemble’s center-to-cardinal exposure ratio.
change: Detect diagonally shifted training samples and randomly center one shifted axis, changing augmentation from 25% center/50% cardinal/25% diagonal to 25% center/75% cardinal/0% diagonal.
mechanism: Cardinalized one-pixel translation augmentation
evidence_used: Restricting translations from ±2 to ±1 improved validation correctness from 9,166 to 9,196, while adding diagonal validation views previously reduced it to 9,155; the prior diagonal-removal attempts timed out and therefore did not provide contrary accuracy evidence.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Eliminating diagonal training translations while preserving balanced one-pixel cardinal shifts will exceed 9,196 correct predictions by aligning augmentation support with the successful validation views.
change: For diagonally shifted samples, deterministically alternate which axis is recentered using batch-index parity, producing 25% centered and 75% balanced cardinal crops without additional random sampling.
mechanism: Low-overhead balanced cardinalization
evidence_used: Restricting translations to ±1 improved correctness from 9,166 to 9,196, while diagonal validation views reduced correctness to 9,155. Earlier diagonal-removal implementations timed out, motivating this computationally lean formulation.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 67.21863225009292, "validation_accuracy": 0.9206, "validation_correct": 9206, "validation_cross_entropy": 0.22713947525024414, "validation_score": 9206.407451646764}

RECENT RESULT
hypothesis: A zero-initialized class-specific log-mean-exp evidence branch will exceed 9,206 correct predictions by learning translation-tolerant local class cues while retaining the position-sensitive flattened classifier.
change: Challenge the assumption that every class decision must pass through one shared dense spatial bottleneck: add per-class 1×1 evidence maps with smooth maximum pooling, and reduce the dense width from 160 to 158 to remain below the parameter ceiling.
mechanism: Additive class-specific spatial evidence pooling
evidence_used: Replacing spatial classification with global context regressed to 9,128, so this preserves the successful flattened head; cardinalized translation augmentation improved correctness to 9,206, making a complementary translation-tolerant class-evidence pathway plausible.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249009, "training_seconds": 76.16207754216157, "validation_accuracy": 0.9193, "validation_correct": 9193, "validation_cross_entropy": 0.23321033401489258, "validation_score": 9193.405445840186}

RECENT RESULT
hypothesis: Cycling through center and cardinal translations in the exact 3:2:2:2:2 validation-view ratio will exceed 9,206 correct predictions by reducing per-batch augmentation sampling noise during the fixed two-pass exposure.
change: Replace random triangular shift generation and diagonal cardinalization with a low-overhead deterministic 11-example cycle containing three centered crops and two crops for each one-pixel cardinal translation.
mechanism: Deterministic validation-matched cardinal augmentation cycle
evidence_used: Removing diagonal translations improved correctness from 9,196 to 9,206; the earlier validation-matched 3:2:2:2:2 proposal timed out, while deterministic balanced cardinalization completed successfully, motivating a deterministic formulation of the previously unverified distribution.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Centering one-eleventh of diagonal candidates before cardinalizing the remainder will exceed 9,206 correct predictions by matching the successful validation ensemble’s 3:8 center-to-cardinal exposure ratio while preserving balanced one-pixel shifts.
change: Convert one-eleventh of sampled diagonal shifts into centered crops using a step-rotated batch-index mask; deterministically cardinalize all remaining diagonals as before.
mechanism: Validation-ratio cardinalization
evidence_used: Eliminating diagonal translations improved correctness from 9,196 to 9,206. The current fast implementation yields a 1:3 center-to-cardinal ratio, while the successful validation ensemble uses 3:8; centering one-eleventh of diagonal candidates closes that specific distribution gap without the costly deterministic-cycle implementation that timed out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249789, "training_seconds": 75.64111574995331, "validation_accuracy": 0.9198, "validation_correct": 9198, "validation_cross_entropy": 0.2278820697784424, "validation_score": 9198.407205229481}



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
