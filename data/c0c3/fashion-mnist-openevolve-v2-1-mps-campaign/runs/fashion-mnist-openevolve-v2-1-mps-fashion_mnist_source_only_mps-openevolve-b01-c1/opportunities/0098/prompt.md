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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 60.70090966694988, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21245321807861328, "validation_score": 9257.412387045162}
prior_hypothesis: Widening the second and residual convolutional stages while shrinking the coordinate-specific classifier will exceed 9,247 correct predictions by learning richer spatial features without adding depth or substantially increasing parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,247 correct predictions by retaining early regularization while sharpening class boundaries as ensemble supervision strengthens late in training.
change: Apply a cosine-decayed label-smoothing value to every individual, pair, and ensemble cross-entropy term.
mechanism: Cosine target hardening
evidence_used: Added flip-consistency regularization reduced correctness from 9,247 to 9,241, while EMA reduced it to 9,225; reducing late-stage target regularization tests whether the established model instead benefits from sharper supervised convergence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding a fifth spatial convolution while minimally shrinking the classifier will exceed 9,247 correct predictions by improving feature extraction within the parameter ceiling.
change: Add a 48-channel residual 3×3 refinement block before flattening and reduce the classifier width from 80 to 79, yielding approximately 248,909 learned parameters.
mechanism: Post-residual spatial refinement
evidence_used: Increasing flattened-head capacity regressed to 9,210 correct, indicating additional capacity is better tested in spatial feature learning than in the classifier; this change preserves the proven stem, augmentation, losses, and ensemble.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 248909, "training_seconds": 56.7227943749167, "validation_accuracy": 0.9228, "validation_correct": 9228, "validation_cross_entropy": 0.21980519981384278, "validation_score": 9228.409901515484}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,247 correct predictions by preserving early regularization while sharpening class boundaries as ensemble supervision strengthens late in training.
change: Compute a cosine-decayed smoothing coefficient and apply it consistently to all individual, pair, and ensemble cross-entropy terms.
mechanism: Cosine target hardening
evidence_used: The prior target-hardening implementation could not be verified, leaving its hypothesis untested; meanwhile added consistency and EMA regularization regressed to 9,241 and 9,225 correct, respectively, motivating reduced late regularization in the established 9,247-correct design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Using batch size 96 with a proportionally moderated learning rate will exceed 9,247 correct predictions by providing 33% more optimizer updates and beneficial gradient noise without increasing total example exposure.
change: Reduce batch size from 128 to 96 and scale the initial and peak learning rates by approximately √(96/128).
mechanism: Smaller-batch, learning-rate-scaled optimization
evidence_used: Recent crop, ensemble, consistency, EMA, and capacity changes all regressed, motivating a shift from invariance and architecture tuning to the still-unexplored optimization granularity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 230442, "training_seconds": 66.7434008331038, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21325289039611817, "validation_score": 9240.412115234967}

RECENT RESULT
hypothesis: Replacing fixed 7×7 flattening with pooled 4×4, 2×2, and global representations will exceed 9,247 correct predictions by preserving coarse garment layout while reducing sensitivity to crop-induced feature displacement.
change: Decode the existing 48-channel feature map through multiscale average pooling and a wider 200-unit classifier, using 245,442 learned parameters.
mechanism: Coarse-to-global spatial pyramid decoder
evidence_used: A larger flattened head regressed to 9,210 and phase-preserving space-to-depth regressed to 9,191, indicating that more fixed-coordinate detail is unhelpful; this efficiently tests hierarchical spatial aggregation without the covariance decoder’s prohibitive runtime.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding lightweight image-conditioned channel gating will exceed 9,247 correct predictions by exploiting global garment context without sacrificing the proven spatial representation or materially increasing runtime.
change: Add a zero-initialized squeeze-and-excitation-style gate after the residual extractor, allowing each image to modulate its 48 feature channels within a bounded range while keeping the model initially identical to the baseline.
mechanism: Identity-initialized global channel recalibration
evidence_used: Additional spatial convolution regressed to 9,228 correct and a larger flattened head regressed to 9,210, while global covariance pooling timed out; this motivates a computationally cheap global-context mechanism that adds neither spatial refinement nor substantial decoder capacity.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 232818, "training_seconds": 66.007294167066, "validation_accuracy": 0.9204, "validation_correct": 9204, "validation_cross_entropy": 0.21860506134033203, "validation_score": 9204.410305205405}

RECENT RESULT
hypothesis: Holding label smoothing at 0.02 for the first half of training and annealing it to zero thereafter will exceed 9,247 correct predictions by preserving early regularization while sharpening decision boundaries during low-learning-rate convergence.
change: Introduce a late-only cosine label-smoothing schedule and apply it consistently to individual-view, flip-pair, and ensemble losses.
mechanism: Late-phase cosine target hardening
evidence_used: Constant regularization remains in the 9,247-correct baseline, while additional consistency and EMA regularization regressed to 9,241 and 9,225; prior target-hardening attempts were not verified, so this hypothesis remains untested.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 56.95399208297022, "validation_accuracy": 0.9242, "validation_correct": 9242, "validation_cross_entropy": 0.2165532070159912, "validation_score": 9242.410997231455}

RECENT RESULT
hypothesis: Applying one shared random rotation of at most 8° per training batch will exceed 9,247 correct predictions by learning modest orientation invariance while preserving the batch-correlated augmentation structure that outperformed per-example transforms.
change: Rotate each training batch by a uniformly sampled angle in [−8°, 8°] with bilinear sampling and reflected boundaries before generating the existing translated and flipped views.
mechanism: Batch-shared mild rotational augmentation
evidence_used: Per-example translation sampling regressed to 9,219 correct while batch-shared offsets remained stronger; recent capacity, EMA, and target-regularization changes also regressed, motivating a new invariance applied with the proven batch-shared structure.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 48.79392929095775, "validation_accuracy": 0.9114, "validation_correct": 9114, "validation_cross_entropy": 0.24383610076904297, "validation_score": 9114.401982222329}

RECENT RESULT
hypothesis: Applying separable 1:2:1 weights only to the central validation crops will exceed 9,247 correct predictions by improving ensemble quality without imposing the training bias that accompanied the jointly weighted result.
change: Weight the central 3×3 validation logits with a normalized separable 1:2:1 kernel while leaving training and the full 5×5 ensemble unchanged.
mechanism: Evaluation-only center-weighted crop aggregation
evidence_used: Joint 1:2:1 weighting reached 9,246 correct but improved cross-entropy from 0.21609 to 0.21532; the evaluation-only variant timed out, so its ability to retain that ensemble benefit without biased crop sampling remains untested.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 50.13438904192299, "validation_accuracy": 0.9247, "validation_correct": 9247, "validation_cross_entropy": 0.21604468002319335, "validation_score": 9247.41116910276}

RECENT RESULT
hypothesis: Adding 10% dropout to the flattened classifier representation will exceed 9,247 correct predictions by reducing reliance on crop-specific coordinate features while preserving inference-time architecture and ensemble behavior.
change: Insert dropout after the classifier’s normalized hidden layer.
mechanism: Low-rate classifier dropout
evidence_used: Increasing flattened-head capacity regressed to 9,210 correct, while additional spatial capacity regressed to 9,228, suggesting the existing representation needs regularization rather than more parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 230442, "training_seconds": 49.79950137506239, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.21660970840454102, "validation_score": 9237.410978144055}

RECENT RESULT
hypothesis: Replacing the fixed 7×7 representation with pooled 4×4, 2×2, and global features will exceed 9,247 correct predictions by retaining garment layout while reducing crop-position sensitivity.
change: Concatenate three average-pooled feature scales and classify them with a 200-unit head, totaling 245,442 learned parameters.
mechanism: Coarse-to-global spatial pyramid decoder
evidence_used: Enlarging the fixed-coordinate head regressed to 9,210 correct, whereas the prior multiscale pooling proposal could not be verified and therefore remains untested.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Widening the second and residual convolutional stages while shrinking the coordinate-specific classifier will exceed 9,247 correct predictions by learning richer spatial features without adding depth or substantially increasing parameters.
change: Change channel widths from 16→32→48 to 16→40→56 and reduce the hidden classifier from 80 to 64, yielding 233,434 learned parameters.
mechanism: Parameter reallocation toward convolutional width
evidence_used: Enlarging the flattened head regressed to 9,210 correct and adding a fifth spatial convolution regressed to 9,228; reallocating capacity from the head into existing convolutional stages tests feature width while avoiding both unsuccessful changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 233434, "training_seconds": 60.70090966694988, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21245321807861328, "validation_score": 9257.412387045162}



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
