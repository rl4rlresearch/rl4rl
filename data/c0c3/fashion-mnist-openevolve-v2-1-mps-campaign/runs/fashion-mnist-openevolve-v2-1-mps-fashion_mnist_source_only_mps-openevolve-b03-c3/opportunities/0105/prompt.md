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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 41.99500587489456, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21068120079040528, "validation_score": 9268.412990636736}
prior_hypothesis: AdamW beta2=0.99 will exceed 9,268 correct predictions by adapting gradient-variance estimates faster during the fixed 522-step training run while retaining the best verified architecture and TTA.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 49.11529295798391, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21245572166442872, "validation_score": 9268.412386193628}
prior_hypothesis: Restoring the verified 40-local/24-context architecture and redistributing the unchanged 0.10 radius-2 TTA weight toward less-displaced axial views will exceed 9,268 correct predictions.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 51.062351166969165, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21246516418457032, "validation_score": 9268.412382982018}
prior_hypothesis: Reallocating eight aggregator channels from the local branch to the successful dilation-2 context branch will exceed 9,265 correct predictions by strengthening broader garment-shape modeling while retaining 40 channels for local detail and keeping parameter count unchanged.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.15950512513518, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21244136581420897, "validation_score": 9268.412391076467}
prior_hypothesis: Favoring vertical over horizontal radius-1 translations while restoring the best radius-2 weights will exceed 9,268 correct predictions because the opposite horizontal bias worsened cross-entropy.

## Recent verification evidence

RECENT RESULT
hypothesis: Reversing the unsuccessful center bias by moving 0.02 ensemble weight from the unshifted view to radius-1 diagonal views will exceed 9,268 correct predictions.
change: Reduce the unshifted TTA weight from 0.36 to 0.34 and increase each radius-1 diagonal weight from 0.045 to 0.050, preserving total ensemble weight and all training settings.
mechanism: Diagonal-favoring inner-translation ensembling
evidence_used: Moving the same 0.02 weight in the opposite direction—from diagonal views to the center—reduced correctness from 9,268 to 9,265, indicating that radius-1 diagonal evidence is more valuable than additional center weight.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 32.745428583119065, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21246579437255858, "validation_score": 9268.41238276768}

RECENT RESULT
hypothesis: Favoring horizontal over vertical radius-1 translations while retaining the best distance-stratified radius-2 weights will exceed 9,268 correct predictions by reducing harmful averaging across vertically informative garment positions.
change: Preserve total TTA weight, increase each horizontal radius-1 shift from 0.09 to 0.10, decrease each vertical shift to 0.08, and restore the radius-2 weighting that tied the best correct count with lower cross-entropy.
mechanism: Axis-aware translation ensembling
evidence_used: Distance-stratified radius-2 TTA preserved 9,268 correct and improved cross-entropy to 0.2124557217; center/diagonal reallocations did not improve correctness, leaving axial orientation as an informative untested dimension.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 48.1495108329691, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.2124831729888916, "validation_score": 9268.412376856964}

RECENT RESULT
hypothesis: Favoring vertical over horizontal radius-1 translations while restoring the best radius-2 weights will exceed 9,268 correct predictions because the opposite horizontal bias worsened cross-entropy.
change: Increase each vertical radius-1 weight to 0.10, decrease each horizontal weight to 0.08, and restore the verified 0.009375/0.00625/0.003125 radius-2 weighting while preserving total ensemble weight.
mechanism: Vertical-axis-aware translation ensembling
evidence_used: Symmetric axis weights with mildly distance-stratified radius-2 TTA achieved 9,268 correct at 0.2124557 cross-entropy; horizontal bias retained 9,268 but worsened cross-entropy to 0.2124832, motivating the opposite directional test.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.15950512513518, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21244136581420897, "validation_score": 9268.412391076467}

RECENT RESULT
hypothesis: Increasing the vertical radius-1 TTA weights to 0.11 and decreasing the horizontal weights to 0.07 will exceed 9,268 correct predictions by extending the cross-entropy improvement observed when moving from horizontal bias to vertical bias.
change: Restore the best 0.36 center and 0.045 diagonal weights, then strengthen the verified vertical-over-horizontal radius-1 weighting while preserving total ensemble weight.
mechanism: Stronger vertical-axis translation ensembling
evidence_used: Reference Design 3’s 0.10 vertical/0.08 horizontal weighting retained 9,268 correct and achieved the best cross-entropy, 0.2124414, while the opposite bias worsened cross-entropy to 0.2124832; this tests whether that directional trend continues.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 39.98342112498358, "validation_accuracy": 0.9263, "validation_correct": 9263, "validation_cross_entropy": 0.21244010124206544, "validation_score": 9263.412391506588}

RECENT RESULT
hypothesis: Matching training augmentation to the successful 0.10 vertical/0.08 horizontal validation weighting will exceed 9,268 correct predictions by learning stronger invariance along the empirically favored axis.
change: Increase each radius-1 vertical training-translation weight from 0.09 to 0.10 and decrease each horizontal weight to 0.08, preserving the augmentation distribution’s total weight and all other settings.
mechanism: Axis-aligned translation augmentation
evidence_used: Moderate vertical-biased TTA retained 9,268 correct and achieved the best cross-entropy, while horizontal bias worsened cross-entropy and stronger vertical bias lost five predictions; this motivates aligning training with the successful moderate bias.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 37.25654145795852, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.2117143325805664, "validation_score": 9252.41263851269}

RECENT RESULT
hypothesis: Vertical/horizontal radius-1 weights of 0.1025/0.0775 will retain 9,268 correct predictions while lowering cross-entropy below 0.2124414, strictly improving validation_score.
change: Extend the successful vertical TTA bias by one quarter of the tested 0.10-to-0.11 interval, preserving total ensemble weight and all training behavior.
mechanism: Conservative vertical-axis TTA interpolation
evidence_used: The 0.10/0.08 weighting retained 9,268 correct and improved cross-entropy to 0.2124414; 0.11/0.07 further reduced cross-entropy to 0.2124401 but lost five predictions, motivating a cautious interpolation near the accuracy-preserving setting.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 49.868663124972954, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.2124398235321045, "validation_score": 9267.412391601047}

RECENT RESULT
hypothesis: Redistributing the successful vertical radius-1 weight from the downward-content view to the upward-content view will exceed 9,268 correct predictions by exploiting directional alignment while preserving the accuracy-safe total vertical weight.
change: Restore the best distance-stratified radius-2 TTA and moderate vertical-over-horizontal weighting, then assign 0.11/0.09 weights to the two opposite vertical shifts instead of 0.10/0.10.
mechanism: Sign-aware vertical translation ensembling
evidence_used: Moderate vertical bias retained 9,268 correct and improved cross-entropy, whereas stronger symmetric vertical bias lost five predictions; this motivates testing vertical directionality without increasing total vertical emphasis.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 40.76835841592401, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.2124018669128418, "validation_score": 9264.412404511775}

RECENT RESULT
hypothesis: Replacing both information-discarding max pools with space-to-depth phase preservation and learned channel mixing will exceed 9,268 correct predictions by retaining subpixel garment structure for the convolutional classifier.
change: Replace the shared max-pooling backbone with PixelUnshuffle-based learned polyphase reductions while retaining the successful spatial classifier, and restore the best verified vertical-biased TTA weights.
mechanism: Learned polyphase downsampling
evidence_used: Head replacement and global channel gating regressed to 9,227 and 9,257 correct while retaining the same max-pooled representation; meanwhile, prediction quality responded to one-pixel TTA orientation, motivating a clean test of whether preserving sampling-phase information before learned mixing improves the representation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 241022, "training_seconds": 43.9022715408355, "validation_accuracy": 0.9207, "validation_correct": 9207, "validation_cross_entropy": 0.2177106430053711, "validation_score": 9207.410606577903}

RECENT RESULT
hypothesis: Favoring the downward-content radius-1 view over the upward-content view will exceed 9,268 correct predictions because the previously tested opposite asymmetry materially changed predictions and reduced cross-entropy, demonstrating useful vertical sign sensitivity but choosing the wrong directional preference.
change: Preserve the accuracy-safe total vertical TTA weight of 0.20 while assigning 0.11 to the downward-content shift and 0.09 to the upward-content shift; retain all other architecture, training, and TTA settings.
mechanism: Opposite sign-aware vertical translation ensembling
evidence_used: Symmetric 0.10/0.10 vertical weighting achieved 9,268 correct, while favoring the upward-content view produced 9,264 with substantially lower 0.2124019 cross-entropy; testing the complementary direction is the cleanest unresolved sign-aware comparison.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 35.01191437500529, "validation_accuracy": 0.9267, "validation_correct": 9267, "validation_cross_entropy": 0.21249540481567383, "validation_score": 9267.412372696848}

RECENT RESULT
hypothesis: Averaging the final 32 low-learning-rate parameter states while retaining the best vertical-biased TTA will exceed Reference Design 3’s validation_score by reducing boundary instability without changing model capacity.
change: Add a tail-averaging AdamW optimizer that installs the mean of the final 32 parameter states before validation, and restore the verified 0.10 vertical/0.08 horizontal TTA weights.
mechanism: Late-trajectory parameter averaging
evidence_used: Reference Design 3 preserved 9,268 correct with the best cross-entropy, while architecture, pooling, gating, and augmentation changes regressed; this motivates preserving its representation and improving the final solution through low-risk trajectory averaging.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 45.657704167068005, "validation_accuracy": 0.9257, "validation_correct": 9257, "validation_cross_entropy": 0.21340101928710936, "validation_score": 9257.412064924994}

RECENT RESULT
hypothesis: Extending the successful moderate vertical preference to radius-2 TTA views will exceed 9,268 correct predictions while preserving total ensemble weight.
change: Restore Reference Design 3’s accuracy-safe radius-1 weights and distance-stratified radius-2 weights, then mildly favor vertically dominant radius-2 translations over horizontally dominant ones.
mechanism: Multiscale vertical-axis translation ensembling
evidence_used: Vertical-biased radius-1 TTA achieved the best verified tied score at 9,268 correct and 0.2124414 cross-entropy, while horizontal bias worsened cross-entropy and stronger radius-1 bias lost accuracy; radius-2 orientation remains untested.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.77745370892808, "validation_accuracy": 0.9265, "validation_correct": 9265, "validation_cross_entropy": 0.21242407264709473, "validation_score": 9265.412396958523}

RECENT RESULT
hypothesis: AdamW beta2=0.99 will exceed 9,268 correct predictions by adapting gradient-variance estimates faster during the fixed 522-step training run while retaining the best verified architecture and TTA.
change: Restore Reference Design 3’s vertical-biased TTA and use a faster-decaying AdamW second-moment estimate.
mechanism: Short-horizon second-moment adaptation
evidence_used: Reference Design 3 achieved the best tied score with 9,268 correct; architecture, augmentation, and late-trajectory averaging changes regressed, motivating a clean optimization-dynamics test on the strongest verified representation.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 41.99500587489456, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.21068120079040528, "validation_score": 9268.412990636736}



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
