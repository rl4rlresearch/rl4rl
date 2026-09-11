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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 67.98884708294645, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.20388637008666993, "validation_score": 9312.415321588835}
prior_hypothesis: Halving the unique-image batch size while adopting the proven two-convolution residual block will exceed Reference Design 2’s 9,310 correct predictions by providing roughly twice as many optimizer updates over the same 100,000 examples.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249393, "training_seconds": 47.79130112496205, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.20765182266235352, "validation_score": 9310.414026618117}
prior_hypothesis: Extending the successful residual refinement into a two-convolution block while remaining below 250,000 parameters will exceed 9,287 correct validation predictions.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249973, "training_seconds": 51.974545875098556, "validation_accuracy": 0.9296, "validation_correct": 9296, "validation_cross_entropy": 0.2071575668334961, "validation_score": 9296.414196136227}
prior_hypothesis: Adding lightweight channel attention to Reference Design 3’s residual features will exceed 9,310 correct validation predictions while remaining below 250,000 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: A 216k-parameter four-convolution model trained with twice as many optimizer updates and horizontal-flip regularization will exceed 8,928 correct validation predictions.
change: Increase convolutional capacity within the parameter ceiling, reduce batch size to 128, add normalization and dropout, apply per-image horizontal flips, use flip-averaged inference, and adopt warmup-cosine AdamW training.
mechanism: Batch-normalized deep CNN with flip invariance
evidence_used: The verified 105,866-parameter shallow model achieved 8,928 correct in only 392 steps; the broader prior change could not be verified, motivating a simpler capacity-and-update increase without affine sampling or residual machinery.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 30.331214583013207, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21675402641296387, "validation_score": 9247.410929398338}

RECENT RESULT
hypothesis: Adding per-image two-pixel translation augmentation to the verified 216,346-parameter flip-invariant CNN will exceed 9,247 correct validation predictions.
change: Adopt Reference Design 1’s architecture and optimization, then add normalized-background-safe random padded crops during training.
mechanism: Translation-regularized flip-invariant CNN
evidence_used: The reference design achieved 9,247 correct versus the current 8,928; translation augmentation is the most focused untested extension of that verified design.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 28.832512624794617, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.2465844955444336, "validation_score": 9141.40109595602}

RECENT RESULT
hypothesis: Optimizing the verified 216,346-parameter model on both horizontal orientations of every presented image, using the same averaged logits used at validation, will exceed 9,247 correct predictions.
change: Adopt Reference Design 1, but replace stochastic one-view flipping with paired original/flip training and an ensemble-level loss.
mechanism: Paired-flip orbit training with flip-ensemble inference
evidence_used: Reference Design 1 achieved the best result at 9,247 correct with flip augmentation and flip-averaged inference, while adding translation reduced performance to 9,141; this motivates strengthening the validated flip invariance without introducing translation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 28.3079188328702, "validation_accuracy": 0.9261, "validation_correct": 9261, "validation_cross_entropy": 0.21184926834106446, "validation_score": 9261.412592566636}

RECENT RESULT
hypothesis: Supervising each horizontal orientation individually while retaining the validation-matched ensemble loss will exceed 9,261 correct predictions.
change: Blend cross-entropy on the averaged logits with cross-entropy on each original and flipped view.
mechanism: Paired-flip deep supervision with ensemble-aligned loss
evidence_used: Paired-flip ensemble training reached 9,261 correct versus 9,247 for stochastic flipping, while translation fell to 9,141; this isolates stronger flip supervision without introducing a harmful augmentation.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 33.73269945802167, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.21099624671936035, "validation_score": 9276.412883195431}

RECENT RESULT
hypothesis: Training both horizontal orientations with ensemble-aligned supervision and evaluating an EMA of the final optimization trajectory will exceed the best prior result of 9,276 correct predictions.
change: Replace stochastic flipping with paired-view training, blend per-view and ensemble losses, and maintain a 0.99-decay parameter EMA that is installed after the final optimizer step.
mechanism: Paired-flip deep supervision with terminal weight averaging
evidence_used: Paired-flip deep supervision achieved 9,276 correct, outperforming stochastic flips at 9,247 and translation augmentation at 9,141; EMA directly extends the strongest verified mechanism by reducing late-step parameter noise without changing its augmentation assumptions.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 216346, "training_seconds": 42.01414312492125, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21310321731567383, "validation_score": 9268.412166081882}

RECENT RESULT
hypothesis: Reallocating dense-head parameters to a residual 7×7 convolution while retaining Reference Design 3’s paired-flip loss will exceed 9,276 correct predictions.
change: Remove harmful translation augmentation, restore paired-flip deep supervision, and replace classifier width with a budget-neutral residual convolutional refinement stage.
mechanism: Budget-neutral residual feature refinement with paired-flip supervision
evidence_used: Reference Design 3 achieved the best result—9,276 correct—with paired-flip supervision, while translation reduced performance to 9,141; its 216,346 parameters are concentrated in the dense head, motivating a clean reallocation toward spatial feature extraction.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 215550, "training_seconds": 37.8976491659414, "validation_accuracy": 0.9287, "validation_correct": 9287, "validation_cross_entropy": 0.2080647979736328, "validation_score": 9287.41388508368}

RECENT RESULT
hypothesis: Extending the successful residual refinement into a two-convolution block while remaining below 250,000 parameters will exceed 9,287 correct validation predictions.
change: Add a second batch-normalized 3×3 refinement convolution with an intermediate GELU, and reduce the dense-head width from 36 to 35, yielding approximately 249,393 learned parameters.
mechanism: Full two-convolution residual spatial refinement
evidence_used: The current single-convolution residual design reached 9,287 correct versus 9,276 for the otherwise similar dense-head design, showing that spatial refinement is beneficial while leaving enough budget for a complete two-layer residual block.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249393, "training_seconds": 47.79130112496205, "validation_accuracy": 0.931, "validation_correct": 9310, "validation_cross_entropy": 0.20765182266235352, "validation_score": 9310.414026618117}

RECENT RESULT
hypothesis: Adding lightweight channel attention to Reference Design 3’s residual features will exceed 9,310 correct validation predictions while remaining below 250,000 learned parameters.
change: Adopt the successful two-convolution residual block and add a 580-parameter squeeze-and-excitation gate initialized around identity scaling, yielding 249,973 learned parameters.
mechanism: Residual channel-attention refinement
evidence_used: Reference Design 3 reached the best result of 9,310 correct after deeper spatial refinement improved on the single-convolution design’s 9,287; its 607-parameter headroom motivates testing targeted channel reweighting without sacrificing proven capacity.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249973, "training_seconds": 51.974545875098556, "validation_accuracy": 0.9296, "validation_correct": 9296, "validation_cross_entropy": 0.2071575668334961, "validation_score": 9296.414196136227}

RECENT RESULT
hypothesis: Halving the unique-image batch size while adopting the proven two-convolution residual block will exceed Reference Design 2’s 9,310 correct predictions by providing roughly twice as many optimizer updates over the same 100,000 examples.
change: Restore the best 249,393-parameter residual architecture and non-EMA schedule, then reduce batch size from 128 to 64.
mechanism: Higher-update residual refinement
evidence_used: Reference Design 2 achieved the best result at 9,310 correct; channel attention reduced this to 9,296 and EMA reduced the earlier architecture to 9,268, motivating retention of the proven model while testing increased update frequency.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 67.98884708294645, "validation_accuracy": 0.9312, "validation_correct": 9312, "validation_cross_entropy": 0.20388637008666993, "validation_score": 9312.415321588835}

RECENT RESULT
hypothesis: Reducing the unique-image batch size from 64 to 32 will exceed 9,312 correct validation predictions by doubling optimizer updates while retaining an effective 64-view paired-flip training batch.
change: Keep the best verified architecture, paired-flip objective, optimizer, and schedule unchanged, and reduce only `BATCH_SIZE` from 64 to 32.
mechanism: Further higher-update residual refinement
evidence_used: Reducing batch size from 128 to 64 improved the same 249,393-parameter implementation from 9,310 to 9,312 correct and lowered cross-entropy from 0.20765 to 0.20389, directly motivating another isolated increase in update frequency.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the unique-image batch size to 48 will exceed 9,312 correct predictions by increasing optimizer updates beyond batch size 64 while remaining fast enough to avoid batch size 32’s verification timeout.
change: Keep the proven architecture, paired-flip objective, and schedule unchanged while reducing `BATCH_SIZE` from 128 to 48.
mechanism: Time-budget-aware intermediate batch scaling
evidence_used: Batch size 64 improved the same implementation from 9,310 to 9,312 correct and reduced cross-entropy from 0.20765 to 0.20389; batch size 32 timed out, motivating an intermediate 48-image batch.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Giving each proven refinement convolution its own identity shortcut at batch size 64 will exceed 9,312 correct predictions by improving gradient flow during the fixed two-pass exposure.
change: Remove harmful channel attention, restore batch size 64, and reorganize the parameter-identical two-convolution refinement into two sequential residual units.
mechanism: Stacked single-convolution residual refinement
evidence_used: The 249,393-parameter batch-64 design achieved the best result at 9,312 correct, while channel attention reduced accuracy to 9,296; earlier evidence also showed that residual spatial refinement improved from 9,287 to 9,310 when deepened.
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
