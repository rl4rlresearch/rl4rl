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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 68.95657641696744, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028644546508789, "validation_score": 9295.415674432865}
prior_hypothesis: BatchNorm1d momentum 0.1978125 will retain 9,295 correct predictions while lowering cross-entropy relative to momentum 0.198125.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 65.31258245906793, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286409759521484, "validation_score": 9295.415674556252}
prior_hypothesis: A classifier BatchNorm1d momentum of 0.197771655718485514323 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 48.46009991597384, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028640983581543, "validation_score": 9295.415674555989}
prior_hypothesis: BatchNorm1d momentum 0.19777164459228515625 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410102844238.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 38.08605529088527, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286409759521484, "validation_score": 9295.415674556252}
prior_hypothesis: A classifier BatchNorm1d momentum of 0.197771655651 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286409759521484.

## Recent verification evidence

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19777099609375 will retain 9,295 correct predictions while lowering validation cross-entropy below 0.20286411476135255.
change: Increase only the classifier BatchNorm1d momentum from 0.1975 to 0.19777099609375.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.1977734375 achieved 9,295 correct with 0.20286411476135255 cross-entropy, while 0.1977685546875 achieved 9,294 correct with lower 0.2028640724182129 cross-entropy; their exact midpoint is the most informative unresolved boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 59.08748662495054, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.20286409683227538, "validation_score": 9294.415674556516}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.197772216796875 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286411476135255.
change: Decrease only the classifier BatchNorm1d momentum from 0.1977734375 to 0.197772216796875.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.1977734375 achieved 9,295 correct with 0.20286411476135255 cross-entropy, while 0.19777099609375 achieved 9,294 correct with lower 0.20286409683227538 cross-entropy; their exact midpoint is the most informative unresolved boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 66.14025454199873, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286410484313966, "validation_score": 9295.415674553748}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.1977716064453125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410484313966.
change: Decrease only the classifier BatchNorm1d momentum from 0.1978125 to 0.1977716064453125.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.197772216796875 achieved 9,295 correct, while 0.19777099609375 achieved 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 57.595671540824696, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.20286409912109374, "validation_score": 9294.415674555725}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19777191162109375 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410484313966.
change: Decrease only the classifier BatchNorm1d momentum from 0.198125 to 0.19777191162109375.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.197772216796875 achieved 9,295 correct, while 0.1977716064453125 achieved 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 56.01320758298971, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286410484313966, "validation_score": 9295.415674553748}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.197771759033203125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410484313966.
change: Increase only the classifier BatchNorm1d momentum from 0.1975 to 0.197771759033203125.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.19777191162109375 achieved 9,295 correct, while 0.1977716064453125 achieved 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 61.377774416003376, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286410102844238, "validation_score": 9295.415674555066}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.1977716827392578125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410102844238.
change: Decrease only the classifier BatchNorm1d momentum from 0.197771759033203125 to 0.1977716827392578125.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.197771759033203125 achieved 9,295 correct with 0.20286410102844238 cross-entropy, while 0.1977716064453125 achieved 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 49.06612583412789, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286410217285156, "validation_score": 9295.41567455467}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19777164459228515625 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286410102844238.
change: Decrease only the classifier BatchNorm1d momentum from 0.197772216796875 to 0.19777164459228515625.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.1977716827392578125 retained 9,295 correct, while 0.1977716064453125 produced 9,294 correct with lower cross-entropy; their exact midpoint is the most informative unresolved correctness-boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 48.46009991597384, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.2028640983581543, "validation_score": 9295.415674555989}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.197771625518798828125 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.
change: Decrease only the classifier BatchNorm1d momentum from 0.19777191162109375 to 0.197771625518798828125.
mechanism: Decision-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.19777164459228515625 achieved 9,295 correct with the best observed cross-entropy, while 0.1977716064453125 produced 9,294 correct with slightly lower cross-entropy; their exact midpoint is the most informative unresolved correctness-boundary test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 71.33113558287732, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286410331726074, "validation_score": 9295.415674554275}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.1977716159820556640625 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.
change: Decrease only the classifier BatchNorm1d momentum from 0.1978125 to 0.1977716159820556640625.
mechanism: Correctness-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.197771625518798828125 retained 9,295 correct, while 0.1977716064453125 produced 9,294 correct; their exact midpoint is the most informative unresolved correctness-boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 55.14363216701895, "validation_accuracy": 0.9294, "validation_correct": 9294, "validation_cross_entropy": 0.20286409912109374, "validation_score": 9294.415674555725}

RECENT RESULT
hypothesis: BatchNorm1d momentum 0.19777162075042724609375 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.
change: Decrease only the classifier BatchNorm1d momentum from 0.19777164459228515625 to 0.19777162075042724609375.
mechanism: Correctness-boundary BatchNorm momentum bisection
evidence_used: Momentum 0.197771625518798828125 retained 9,295 correct, while 0.1977716159820556640625 produced 9,294; their exact midpoint is the most informative unresolved correctness-boundary test.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 74.20555308414623, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286410331726074, "validation_score": 9295.415674554275}

RECENT RESULT
hypothesis: A classifier BatchNorm1d momentum of 0.197771655718485514323 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.2028640983581543.
change: Change only the classifier BatchNorm1d momentum to the estimated local cross-entropy minimum.
mechanism: Three-point parabolic BatchNorm momentum interpolation
evidence_used: Momentum 0.19777164459228515625 produced the best observed cross-entropy, while the bracketing 0.197771625518798828125 and 0.1977716827392578125 values retained 9,295 correct with higher cross-entropies; parabolic interpolation of those three results estimates a minimum slightly above the current best momentum.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 65.31258245906793, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286409759521484, "validation_score": 9295.415674556252}

RECENT RESULT
hypothesis: A classifier BatchNorm1d momentum of 0.197771655651 will retain 9,295 correct predictions while reducing validation cross-entropy below 0.20286409759521484.
change: Change only the classifier BatchNorm1d momentum to the refined local cross-entropy minimum.
mechanism: Refined three-point parabolic BatchNorm momentum interpolation
evidence_used: Momentum 0.197771655718485514323 achieved the best observed cross-entropy; the higher-loss bracketing results at 0.19777164459228515625 and 0.1977716827392578125 imply a fitted minimum approximately 6.7e-11 below it.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248682, "training_seconds": 38.08605529088527, "validation_accuracy": 0.9295, "validation_correct": 9295, "validation_cross_entropy": 0.20286409759521484, "validation_score": 9295.415674556252}



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
