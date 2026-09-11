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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.06414550007321, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23744633865356446, "validation_score": 9192.404057925085}
prior_hypothesis: Weighting diagonal views at 0.75 will retain the extra correct prediction of uniform 3×3 averaging while lowering cross-entropy, producing a validation score above 9191.404013889833.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 41.042545708827674, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742110137939454, "validation_score": 9191.40406616587}
prior_hypothesis: A diagonal weight of 0.70 will retain the 9,192 correct predictions achieved at 0.75 while lowering cross-entropy by moving toward the better-calibrated but less accurate 0.50 ensemble.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 66.80932720797136, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23758121337890625, "validation_score": 9191.404013889833}
prior_hypothesis: Adding the four missing diagonal one-pixel views and their flips will exceed 9,190 correct predictions by further reducing residual diagonal-shift sensitivity.

## Recent verification evidence

RECENT RESULT
hypothesis: Increasing convolutional capacity while training on random translations and flips will exceed 8,928 correct predictions within the fixed exposure and parameter budgets.
change: Use a 241,538-parameter four-convolution network, batch size 128, spatial augmentation, lighter label smoothing, cosine AdamW, and horizontal-flip evaluation averaging.
mechanism: Invariance-trained deeper batch-normalized CNN with flip test-time averaging
evidence_used: The current 105,866-parameter two-convolution model reaches 89.28% accuracy, suggesting additional feature-extraction capacity and explicit spatial invariance are promising under the 250,000-parameter ceiling.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241538, "training_seconds": 36.692127916030586, "validation_accuracy": 0.9168, "validation_correct": 9168, "validation_cross_entropy": 0.24475890426635744, "validation_score": 9168.40168421233}

RECENT RESULT
hypothesis: Replacing the reference design’s large dense head with two additional residual convolutions while retaining its augmentation and optimization recipe will exceed 9,168 correct validation predictions within the parameter budget.
change: Use a 249,260-parameter six-convolution residual CNN with batch size 128, translation/flip augmentation, flip-averaged evaluation, light label smoothing, and cosine AdamW.
mechanism: Residual convolutional capacity reallocation
evidence_used: The 241,538-parameter augmented four-convolution reference achieved 91.68% accuracy; most of its parameters are concentrated in the dense head, motivating reallocating them to deeper spatial feature extraction with residual shortcuts.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 51.643368375021964, "validation_accuracy": 0.9177, "validation_correct": 9177, "validation_cross_entropy": 0.24093204765319826, "validation_score": 9177.40292294888}

RECENT RESULT
hypothesis: Adding four one-pixel translation views to the best 249,260-parameter residual design’s flip ensemble will exceed 9,177 correct predictions by reducing residual shift sensitivity without changing training exposure or learned-parameter count.
change: Adopt Reference Design 2’s residual CNN and training recipe, then average validation logits across centered, four cardinally shifted, and horizontally flipped views.
mechanism: Matched translation-and-flip test-time ensembling
evidence_used: Reference Design 2 achieved 9,177 correct with random ±2-pixel translations but evaluates only the centered image and its flip, motivating an evaluation ensemble matched to its learned translation invariance.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 65.1923690000549, "validation_accuracy": 0.919, "validation_correct": 9190, "validation_cross_entropy": 0.23743358840942383, "validation_score": 9190.40406208841}

RECENT RESULT
hypothesis: Adding the four missing diagonal one-pixel views and their flips will exceed 9,190 correct predictions by further reducing residual diagonal-shift sensitivity.
change: Expand evaluation from five spatial positions and their flips to the complete 3×3 one-pixel translation grid and its flipped views.
mechanism: Full 3×3 translation-and-flip logit ensemble
evidence_used: Cardinal-shift-and-flip ensembling improved the same residual model from 9,177 to 9,190 correct while lowering cross-entropy, directly supporting broader matched translation averaging.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 66.80932720797136, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23758121337890625, "validation_score": 9191.404013889833}

RECENT RESULT
hypothesis: Averaging all 25 translations from the same ±2-pixel distribution used during training, together with their flips, will exceed 9,191 correct predictions by reducing residual shift variance.
change: Replace two-view evaluation with uniform logit averaging over the complete 5×5 translation grid and both horizontal orientations.
mechanism: Full augmentation-support translation-and-flip ensemble
evidence_used: Expanding flip evaluation to cardinal one-pixel shifts improved correct predictions from 9,177 to 9,190, and completing the 3×3 grid reached 9,191; the training distribution spans the larger ±2-pixel grid proposed here.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 49.98154616705142, "validation_accuracy": 0.9188, "validation_correct": 9188, "validation_cross_entropy": 0.24227803649902344, "validation_score": 9188.402486388159}

RECENT RESULT
hypothesis: Using the best residual 3×3 TTA design while restricting training translations to the same ±1-pixel support will exceed 9,191 correct predictions by avoiding harmful extreme shifts.
change: Replace the current CNN with the 249,260-parameter residual network and full 3×3 translation/flip ensemble, then narrow random training translations from ±2 to ±1 pixels.
mechanism: Matched one-pixel translation invariance
evidence_used: The residual design with uniform 3×3 TTA achieved 9,191 correct, while expanding evaluation to the full ±2 support fell to 9,188, suggesting two-pixel shifts can dilute useful predictions and motivating matched narrower augmentation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Giving diagonal views half weight will exceed 9,191 correct predictions by retaining their useful corrections while reducing the cross-entropy degradation observed with uniform 3×3 averaging.
change: Keep centered and cardinal translation logits at full weight, but down-weight the four diagonal translations and their flips by 50%.
mechanism: Shrinkage-weighted translation-and-flip ensemble
evidence_used: Adding diagonal views raised correct predictions from 9,190 to 9,191 but worsened cross-entropy from 0.237434 to 0.237581, indicating that diagonal evidence is useful yet noisier than centered and cardinal views.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.60364616592415, "validation_accuracy": 0.919, "validation_correct": 9190, "validation_cross_entropy": 0.23733386154174804, "validation_score": 9190.404094655081}

RECENT RESULT
hypothesis: Weighting diagonal views at 0.75 will retain the extra correct prediction of uniform 3×3 averaging while lowering cross-entropy, producing a validation score above 9191.404013889833.
change: Replace the harmful 5×5 evaluation ensemble with a 3×3 translation-and-flip ensemble that gives centered and cardinal views full weight and diagonal views three-quarter weight.
mechanism: Three-quarter-weight diagonal translation ensemble
evidence_used: Uniform 3×3 averaging achieved 9,191 correct, while half-weight diagonals reduced cross-entropy but lost one correct prediction; an intermediate weight tests whether the accuracy gain can be retained with less noisy diagonal influence.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.06414550007321, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23744633865356446, "validation_score": 9192.404057925085}

RECENT RESULT
hypothesis: A diagonal weight of 0.70 will retain the 9,192 correct predictions achieved at 0.75 while lowering cross-entropy by moving toward the better-calibrated but less accurate 0.50 ensemble.
change: Replace the current CNN with the 249,260-parameter residual design and use a full 3×3 translation-and-flip ensemble with diagonal views weighted at 0.70.
mechanism: Conservative diagonal-view shrinkage
evidence_used: Diagonal weights of 0.50, 0.75, and 1.00 produced 9,190, 9,192, and 9,191 correct respectively, while lower weights reduced cross-entropy; a small reduction from 0.75 targets the observed accuracy/calibration boundary.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 41.042545708827674, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.23742110137939454, "validation_score": 9191.40406616587}



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
