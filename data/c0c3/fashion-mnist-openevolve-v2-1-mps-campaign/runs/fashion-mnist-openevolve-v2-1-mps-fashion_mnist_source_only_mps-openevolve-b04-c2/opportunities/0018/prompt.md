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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.08498666691594, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20959415397644043, "validation_score": 9273.413361786146}
prior_hypothesis: Restoring the best plain pairwise-refinement model and multiplying validation logits by 1.1 will preserve its 9,273 correct predictions while reducing cross-entropy below 0.2151.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.11338229198009, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.21639883575439453, "validation_score": 9270.411049390466}
prior_hypothesis: The best verified pairwise-refinement model with a 10× lower terminal learning rate will exceed 9,273 correct predictions by allowing more precise late-stage convergence.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 36.3837982499972, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.21509943618774413, "validation_score": 9273.41148895729}
prior_hypothesis: An unaugmented batch-64 model with 5×5 pairwise grouped refinement will exceed Reference Design 3’s 9,258 correct predictions while remaining below 250,000 parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Adding pointwise channel mixing after the depthwise convolution while narrowing the dense hidden layer will exceed 9,258 correct predictions without exceeding 250,000 parameters.
change: Insert a batch-normalized 1×1 convolution after depthwise refinement and reduce the classifier width from 128 to 112, preserving all verified training settings and using approximately 241,210 learned parameters.
mechanism: Depthwise-separable spatial refinement with post-filter channel mixing
evidence_used: The unaugmented 243,178-parameter design achieved 9,258 correct, while horizontal flips and translations reduced accuracy; this motivates retaining its verified optimization path and reallocating dense-layer capacity to more expressive learned features.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 241210, "training_seconds": 44.320714874891564, "validation_accuracy": 0.9246, "validation_correct": 9246, "validation_cross_entropy": 0.2186981990814209, "validation_score": 9246.410273848256}

RECENT RESULT
hypothesis: The verified unaugmented 243,178-parameter CNN trained with batch size 64 and approximately 1,563 optimizer steps will exceed 9,258 correct validation predictions.
change: Remove flip training and flip-ensemble inference while halving batch size, preserving the qualified architecture, loss, optimizer, and exposure-normalized schedule.
mechanism: Smaller-batch update densification without geometric augmentation
evidence_used: The unaugmented batch-128 design achieved 9,258 correct, while horizontal flips reduced this to 9,230 and translations to 9,102; batch 64 therefore tests additional optimizer updates without repeating the harmful augmentation changes.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 243178, "training_seconds": 79.40342833288014, "validation_accuracy": 0.9258, "validation_correct": 9258, "validation_cross_entropy": 0.21082799301147462, "validation_score": 9258.412940568673}

RECENT RESULT
hypothesis: An unaugmented batch-64 model with 5×5 pairwise grouped refinement will exceed Reference Design 3’s 9,258 correct predictions while remaining below 250,000 parameters.
change: Restore the best verified batch size and unaugmented training path, then replace the 3×3 depthwise layer with a 5×5 groups-64 convolution, retaining the 128-wide classifier and using approximately 248,426 parameters.
mechanism: Pairwise 5×5 grouped spatial refinement
evidence_used: Reference Design 3 achieved 9,258 correct and the best cross-entropy; translations and flips reduced accuracy, while pointwise mixing with a narrower classifier reached only 9,246. This motivates preserving the verified classifier and spending remaining capacity on coupled spatial refinement.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 36.3837982499972, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.21509943618774413, "validation_score": 9273.41148895729}

RECENT RESULT
hypothesis: Increasing each refinement group from two to four channels while retaining a 122-wide classifier will exceed the current 9,273 correct predictions within the parameter ceiling.
change: Change the 5×5 refinement convolution from 64 to 32 groups and reduce the classifier width from 128 to 122, yielding approximately 247,848 parameters while preserving the verified batch-64 training procedure.
mechanism: Four-channel 5×5 grouped spatial refinement
evidence_used: Pairwise 5×5 refinement improved validation-correct from 9,258 to 9,273. This patch tests further channel coupling while avoiding the prior pointwise-mixing design’s larger classifier reduction to 112 units.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 247848, "training_seconds": 53.60447679203935, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.21435786209106444, "validation_score": 9244.41174024199}

RECENT RESULT
hypothesis: Adding an identity path around the verified pairwise 5×5 refinement will exceed 9,273 correct predictions by preserving established features while learning spatial corrections.
change: Replace the depthwise 3×3 stage with a residual 5×5 groups-64 refinement block, preserving the best batch-64 optimizer, loss, schedule, classifier width, and 248,426-parameter count.
mechanism: Residual pairwise 5×5 spatial refinement
evidence_used: Pairwise 5×5 refinement improved validation-correct from 9,258 to 9,273, while increasing coupling to four channels per group reduced it to 9,244; retaining pairwise grouping and adding a parameter-free identity path isolates a less destructive refinement mechanism.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 48.181277624797076, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21659127655029298, "validation_score": 9268.410984370543}

RECENT RESULT
hypothesis: The best verified pairwise-refinement model with a 10× lower terminal learning rate will exceed 9,273 correct predictions by allowing more precise late-stage convergence.
change: Restore the verified batch-64, groups-64 5×5 architecture and reduce the cosine schedule’s minimum learning rate from 1e-4 to 1e-5.
mechanism: Pairwise 5×5 refinement with deeper cosine annealing
evidence_used: Reference Design 3 achieved the best result at 9,273 correct, while update densification improved cross-entropy; its 0.2151 cross-entropy leaves room for a lower terminal rate to refine the final solution.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.11338229198009, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.21639883575439453, "validation_score": 9270.411049390466}

RECENT RESULT
hypothesis: Evaluating a 0.99-decay moving average of the best model’s training trajectory will exceed 9,273 correct predictions by reducing noise from its 1,564 small-batch updates without altering its successful architecture or learning-rate schedule.
change: Preserve the current model and training procedure, maintain an exponential moving average of every learned parameter after optimizer steps, and install those averaged parameters after the final step for validation.
mechanism: Exponential moving-average checkpoint smoothing
evidence_used: The current pairwise-refinement model is best at 9,273 correct, while residual refinement and a 10× lower terminal learning rate fell to 9,268 and 9,270; this motivates smoothing the verified trajectory rather than changing its features or endpoint schedule again.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 44.63960437499918, "validation_accuracy": 0.9269, "validation_correct": 9269, "validation_cross_entropy": 0.2148154499053955, "validation_score": 9269.411585150681}

RECENT RESULT
hypothesis: Replacing the single pairwise 5×5 refinement with two pairwise 3×3 stages will exceed 9,273 correct predictions by preserving its effective 5×5 receptive field while adding an intermediate nonlinearity.
change: Restore the best verified 1e-4 terminal learning rate and factor the 5×5 groups-64 refinement into two batch-normalized 3×3 groups-64 stages, reducing parameters to approximately 246,890.
mechanism: Two-stage nonlinear pairwise refinement
evidence_used: Single-stage pairwise refinement achieved the best result at 9,273 correct, while stronger four-channel coupling fell to 9,244 and residual refinement reached 9,268; this motivates retaining non-residual pairwise coupling while testing additional nonlinear depth.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 246890, "training_seconds": 59.85714500001632, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21580959205627442, "validation_score": 9245.411248606088}

RECENT RESULT
hypothesis: Restoring the best non-residual pairwise 5×5 refinement and replacing its final non-overlapping pool with a 3×3 stride-2 pool will exceed 9,273 correct predictions by retaining information from all 7×7 feature positions.
change: Replace the residual refinement with the verified plain groups-64 convolution and use overlapping final max pooling, while preserving the optimizer, loss, schedule, batch size, and 248,426-parameter count.
mechanism: Full-coverage overlapping spatial pooling
evidence_used: Plain pairwise refinement achieved the best 9,273 correct, while residual refinement fell to 9,268 and two-stage refinement to 9,245. Its final 2×2 stride-2 pool maps 7×7 to 3×3 while ignoring the last row and column; 3×3 stride-2 pooling covers the full map without adding parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 46.623481875052676, "validation_accuracy": 0.9259, "validation_correct": 9259, "validation_cross_entropy": 0.21919828262329102, "validation_score": 9259.410105564555}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,273 correct predictions by retaining early regularization while allowing hard-label margin refinement late in training.
change: Preserve the best architecture, optimizer, schedule, and batch size; replace constant label smoothing with cosine-decayed smoothing.
mechanism: Annealed target sharpening
evidence_used: The current design is best at 9,273 correct, while residual refinement, deeper cosine annealing, EMA, and alternate pooling all regressed, motivating an isolated loss-target change instead of another architectural or learning-rate modification.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 48.364963749889284, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.20777745056152344, "validation_score": 9244.41398355282}

RECENT RESULT
hypothesis: Removing the classifier’s mild dropout while restoring the verified 1e-4 terminal learning rate will exceed 9,273 correct predictions by improving limited-exposure feature utilization while retaining label smoothing, weight decay, and batch normalization as regularizers.
change: Restore the best verified cosine minimum learning rate and replace the classifier dropout with an identity operation, preserving architecture dimensions, parameter count, batch size, optimizer, and loss.
mechanism: Dropout-free classifier with restored best cosine endpoint
evidence_used: The plain pairwise 5×5 model achieved the best result of 9,273 correct with a 1e-4 terminal rate; lowering it to 1e-5 fell to 9,270, and annealing label smoothing fell to 9,244. This motivates restoring both verified settings and testing the previously unchanged classifier dropout in isolation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.65080608311109, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.21516940689086914, "validation_score": 9264.411465263333}

RECENT RESULT
hypothesis: Restoring the best plain pairwise-refinement model and multiplying validation logits by 1.1 will preserve its 9,273 correct predictions while reducing cross-entropy below 0.2151.
change: Remove the unsuccessful residual identity path, restore non-residual 5×5 groups-64 refinement, and apply fixed positive logit scaling only in evaluation mode.
mechanism: Evaluation-only logit sharpening
evidence_used: Reference Design 3 achieved the best 9,273 correct predictions; annealed target sharpening lowered cross-entropy to 0.2078 but changed decisions. Evaluation-only scaling sharpens confidence without changing argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.08498666691594, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20959415397644043, "validation_score": 9273.413361786146}



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
