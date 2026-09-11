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
verified_results: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 53.9172216670122, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21459944610595702, "validation_score": 9240.411658346793}
prior_hypothesis: Raising the peak learning rate from 3.0e-3 to 3.3e-3 will exceed 9,239 correct predictions by approximating the greater cumulative optimization progress sought with batch size 88, without its runtime increase.

## Recent verification evidence

RECENT RESULT
hypothesis: Restricting training translations from ±2 pixels to ±1 will exceed 9,162 correct predictions by matching the successful validation views and reducing exposure spent on strongly displaced images.
change: Change random training crops from the five-by-five ±2 translation grid to the three-by-three ±1 grid; retain flips, optimization, architecture, and center-weighted inference.
mechanism: Augmentation-matched one-pixel translation training
evidence_used: One-pixel cardinal inference improved correctness from 9,134 to 9,159, while adding more displaced inference influence regressed and center weighting improved the result to 9,162; this suggests moderate translation robustness helps but excessive displacement is harmful.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 42.766273458022624, "validation_accuracy": 0.9175, "validation_correct": 9175, "validation_cross_entropy": 0.22528525276184083, "validation_score": 9175.408068242781}

RECENT RESULT
hypothesis: Reallocating the location-sensitive flattened head into a 96-channel local/dilated stage and pooled coarse-shape/saliency head will exceed 9,175 correct predictions by learning more discriminative garment-scale structure.
change: Add a parallel standard/dilated residual block, widen the final representation from 64 to 96 channels, and replace the 7×7 flattening bottleneck with 2×2 average plus global-max pooling.
mechanism: Parallel local/dilated residual features with spatial-pyramid classification
evidence_used: Augmentation and inference refinements plateaued near 9,175 correct, while the current model spends 94,420 parameters compressing exact 7×7 positions into only 30 features. This patch instead uses 247,142 parameters to learn local and larger-context features while retaining coarse layout and translation-tolerant saliency.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sampling only centered and cardinal one-pixel translations, with the center sampled twice, will exceed 9,175 correct predictions by removing harmful diagonal exposure and matching the successful center-weighted validation ensemble.
change: Replace uniform sampling over all nine ±1 translations with a six-entry distribution containing two centered and four cardinal offsets.
mechanism: Inference-distribution-matched geometric augmentation
evidence_used: Restricting training from ±2 to ±1 improved correctness from 9,162 to 9,175; diagonal validation views regressed, while doubling the centered-view weight improved inference, motivating the same center/cardinal distribution during training.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 245044, "training_seconds": 50.12612995808013, "validation_accuracy": 0.9204, "validation_correct": 9204, "validation_cross_entropy": 0.21821519927978517, "validation_score": 9204.41043651425}

RECENT RESULT
hypothesis: Adding lightweight channel-attention gates to every residual block will exceed 9,204 correct predictions by improving feature selection while preserving the verified architecture, augmentation, and optimization behavior.
change: Add identity-initialized squeeze-and-excitation gates to all residual blocks, increasing learned parameters from 245,044 to 247,528 without changing exposure, batch size, or inference views.
mechanism: Identity-centered squeeze-and-excitation residual recalibration
evidence_used: The current topology reaches 9,204 correct and leaves 4,956 parameters unused, while the prior large architectural redesign could not be verified; a 2,484-parameter, topology-preserving attention addition is a lower-risk test of additional representational capacity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing batch size from 128 to 96 will exceed 9,204 correct predictions by increasing optimizer updates from 782 to about 1,042 while avoiding the runtime cost that caused batch size 64 to time out.
change: Change only the training batch size from 128 to 96, preserving the verified architecture, matched center/cardinal augmentation, schedule, and inference ensemble.
mechanism: Intermediate smaller-batch optimization
evidence_used: The 9,204-correct design is the strongest verified implementation; batch size 64 was motivated by additional optimizer updates but timed out, so 96 is the most informative intermediate runtime–update tradeoff.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 70.92706137499772, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.21391976051330566, "validation_score": 9239.411888838344}

RECENT RESULT
hypothesis: Reducing batch size from 96 to 88 will exceed 9,239 correct predictions by increasing optimizer steps from 1,042 to about 1,137 while remaining substantially closer to the verified runtime than the timed-out batch-size-64 design.
change: Change only the training batch size from 96 to 88, preserving the architecture, optimizer, schedule, matched center/cardinal augmentation, and inference ensemble.
mechanism: Moderately smaller-batch optimization
evidence_used: Reducing batch size from 128 to 96 improved validation correctness from 9,204 to 9,239, whereas 64 timed out; 88 is a conservative continuation of the demonstrated update-frequency benefit.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising the peak learning rate from 3.0e-3 to 3.3e-3 will exceed 9,239 correct predictions by approximating the greater cumulative optimization progress sought with batch size 88, without its runtime increase.
change: Increase the learning-rate schedule uniformly by 10% while retaining batch size 96 and all verified architecture, augmentation, and inference behavior.
mechanism: Update-budget-compensated learning-rate scaling
evidence_used: Reducing batch size from 128 to 96 improved correctness from 9,204 to 9,239 through more optimizer updates, while batch size 88 timed out; scaling 3.0e-3 by the attempted update-count ratio of approximately 1,137/1,042 gives about 3.27e-3, motivating the conservative 3.3e-3 value.
result: improved the objective and became an available design
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 53.9172216670122, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.21459944610595702, "validation_score": 9240.411658346793}

RECENT RESULT
hypothesis: Widening the flattened classifier bottleneck from 30 to 31 features will exceed 9,240 correct predictions by using additional representational capacity without the runtime overhead of convolutional attention.
change: Increase both classifier linear layers to a 31-feature hidden width, raising learned parameters from 245,044 to 248,191.
mechanism: Compute-neutral classifier bottleneck widening
evidence_used: The strongest design leaves 4,956 parameters unused, while the 2,484-parameter squeeze-and-excitation addition timed out; widening the existing head uses 3,147 of those parameters with negligible added computation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing the cosine schedule floor from 5% to 1% will exceed 9,240 correct predictions by retaining the successful 3.3e-3 peak while reducing late-stage update noise.
change: Lower the terminal learning-rate multiplier from 0.05 to 0.01 without changing runtime, batch size, augmentation, or architecture.
mechanism: Lower-floor cosine consolidation
evidence_used: Raising the peak learning rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened validation cross-entropy from 0.21392 to 0.21460, suggesting useful early optimization paired with insufficient late-stage settling.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Reducing warmup from 5% to 2% will exceed 9,240 correct predictions by providing more near-peak-rate updates while preserving the successful 3.3e-3 peak and late cosine decay.
change: Shorten the cosine schedule’s warmup period from about 52 optimizer steps to about 21, with no change to runtime, architecture, augmentation, or inference.
mechanism: Shortened learning-rate warmup
evidence_used: Reducing batch size from 128 to 96 improved correctness from 9,204 to 9,239 through additional optimization updates, while raising peak LR to 3.3e-3 gained another correct prediction; shorter warmup tests additional early optimization progress without the timeout risk of batch size 88 or a still-higher peak rate.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 65.60579987498932, "validation_accuracy": 0.921, "validation_correct": 9210, "validation_cross_entropy": 0.22230840530395507, "validation_score": 9210.409062064722}

RECENT RESULT
hypothesis: Setting the peak learning rate to 3.15e-3 will exceed 9,240 correct predictions by retaining most of the optimization benefit of 3.3e-3 while reducing its apparent late-stage overshoot.
change: Reduce only the peak learning rate from 3.3e-3 to the midpoint between the two verified rates, preserving the successful batch size, warmup, architecture, augmentation, and inference ensemble.
mechanism: Interpolated peak-rate calibration
evidence_used: At batch size 96, increasing the peak rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened cross-entropy from 0.21392 to 0.21460; the midpoint directly tests whether the accuracy gain can be retained with better convergence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 245044, "training_seconds": 66.90865995781496, "validation_accuracy": 0.9227, "validation_correct": 9227, "validation_cross_entropy": 0.21553845291137697, "validation_score": 9227.411340339586}

RECENT RESULT
hypothesis: Replacing the fixed-position 30-feature bottleneck with seven learned spatial part summaries plus a global summary will exceed 9,240 correct predictions by preserving substantially more channel information while making prediction less dependent on exact 7×7 coordinates.
change: Replace the 94,420-parameter flattened classifier with a 235,793-total-parameter attention-pooling model that dynamically locates seven discriminative regions, retains a global feature token, and classifies their concatenated 512-feature representation.
mechanism: Content-addressed multi-part attention pooling
evidence_used: The strongest gains came from translation-matched training and translated-view inference, indicating that limited position invariance is valuable, while the current head remains load-bearing on exact spatial coordinates and compresses 3,136 activations to only 30. This tests the alternative assumption that content-addressed garment parts are more useful than a fixed-location bottleneck without the convolutional overhead that caused attention-gated and widened designs to time out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 96, "examples_processed": 100000, "optimizer_steps": 1042, "parameters": 235793, "training_seconds": 67.15439787507057, "validation_accuracy": 0.9228, "validation_correct": 9228, "validation_cross_entropy": 0.21531134490966797, "validation_score": 9228.411417207693}



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
