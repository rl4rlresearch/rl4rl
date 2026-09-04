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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 232778, "training_seconds": 67.78072387492284, "validation_accuracy": 0.9024, "validation_correct": 9024, "validation_cross_entropy": 0.25752065048217776, "validation_score": 9024.397607784658}
prior_hypothesis: A 232,778-parameter two-block residual CNN trained with 64-image batches and mild translation augmentation will exceed the reference design’s 9,141 correct predictions by shifting capacity from its large dense layer into spatial feature extraction and providing roughly twice as many optimizer updates.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241274, "training_seconds": 41.21943400008604, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.2078806953430176, "validation_score": 9236.413948167172}
prior_hypothesis: Extending the 9,214-correct Reference Design 3 with a 32-channel bottleneck residual block at 7×7 resolution will exceed 9,214 correct predictions while retaining its successful batch-128 training, classifier capacity, augmentation, and flip ensemble.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 227706, "training_seconds": 33.428059292025864, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.21726366081237794, "validation_score": 9214.410757353642}
prior_hypothesis: Adding a parameter-efficient 32-channel residual block before the first pooling layer while preserving the verified batch-128 training and flip ensemble will exceed 9,161 correct predictions.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing capacity to 209,146 parameters, adding residual normalization, and using 128-example batches will exceed the current 8,928 correct predictions within the same 100,000-example exposure budget.
change: Replace the shallow CNN with a three-convolution residual network, halve batch size, add random horizontal flips, remove label smoothing, relax clipping, and use cosine-decayed AdamW.
mechanism: Residual batch-normalized CNN with higher update frequency and symmetry augmentation
evidence_used: The starting 105,866-parameter model reaches 89.28% accuracy after only 392 updates, suggesting limited representation and optimization steps are the primary opportunities while substantial parameter headroom remains.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 209146, "training_seconds": 32.71285529085435, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.23497278366088867, "validation_score": 9141.404867221865}

RECENT RESULT
hypothesis: A 232,778-parameter two-block residual CNN trained with 64-image batches and mild translation augmentation will exceed the reference design’s 9,141 correct predictions by shifting capacity from its large dense layer into spatial feature extraction and providing roughly twice as many optimizer updates.
change: Add a second residual block, reduce the dense head width, use batch size 64, introduce random two-pixel translations, and retune AdamW’s learning rate for the smaller batch.
mechanism: Deeper residual feature learning with parameter-efficient head and translated crops
evidence_used: The 209,146-parameter residual reference reached 91.41% after 782 updates, substantially outperforming the shallow model; its 150,528-weight first dense layer leaves room to deepen convolutional processing while remaining below the parameter ceiling.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 232778, "training_seconds": 67.78072387492284, "validation_accuracy": 0.9024, "validation_correct": 9024, "validation_cross_entropy": 0.25752065048217776, "validation_score": 9024.397607784658}

RECENT RESULT
hypothesis: Retaining the verified 209,146-parameter, batch-128 residual design while averaging original and mirrored validation predictions will exceed 9,141 correct predictions because its training distribution already includes 50% horizontal flips.
change: Adopt Reference Design 1’s architecture and training regimen, then add parameter-free horizontal-flip probability averaging during evaluation.
mechanism: Training-aligned horizontal-flip test-time ensemble
evidence_used: Reference Design 1 achieved 9,141 correct predictions, while the deeper batch-64 design with translation fell to 9,024; this preserves the winning capacity and optimization choices and exploits its already-trained flip invariance without adding training noise.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 209146, "training_seconds": 21.856981082819402, "validation_accuracy": 0.9161, "validation_correct": 9161, "validation_cross_entropy": 0.23078228607177734, "validation_score": 9161.406245690776}

RECENT RESULT
hypothesis: Adding a parameter-efficient 32-channel residual block before the first pooling layer while preserving the verified batch-128 training and flip ensemble will exceed 9,161 correct predictions.
change: Add two normalized 3×3 convolutions in a residual block at 28×28 resolution, increasing capacity from 209,146 to approximately 227,706 parameters without altering the successful optimizer, augmentation, classifier, or evaluation ensemble.
mechanism: High-resolution residual feature refinement
evidence_used: The 209,146-parameter residual design reached 9,161 correct, while the 232,778-parameter experiment fell to 9,024 after simultaneously shrinking its classifier, changing batch size, and adding translations; this patch isolates additional spatial depth while retaining every verified winning training choice.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 227706, "training_seconds": 33.428059292025864, "validation_accuracy": 0.9214, "validation_correct": 9214, "validation_cross_entropy": 0.21726366081237794, "validation_score": 9214.410757353642}

RECENT RESULT
hypothesis: Extending the 9,214-correct Reference Design 3 with a 32-channel bottleneck residual block at 7×7 resolution will exceed 9,214 correct predictions while retaining its successful batch-128 training, classifier capacity, augmentation, and flip ensemble.
change: Adopt Reference Design 3’s high-resolution residual block and evaluation ensemble, then add a 64→32→32→64 bottleneck residual block before the classifier, bringing the model to approximately 241,274 learned parameters.
mechanism: Low-resolution bottleneck residual refinement
evidence_used: Reference Design 3 improved from 9,161 to 9,214 correct by adding residual spatial refinement while preserving the proven regimen; the bottleneck adds further spatial processing within the parameter ceiling without the classifier shrinkage, batch-size change, or translation augmentation that confounded Reference Design 2.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241274, "training_seconds": 41.21943400008604, "validation_accuracy": 0.9236, "validation_correct": 9236, "validation_cross_entropy": 0.2078806953430176, "validation_score": 9236.413948167172}



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
