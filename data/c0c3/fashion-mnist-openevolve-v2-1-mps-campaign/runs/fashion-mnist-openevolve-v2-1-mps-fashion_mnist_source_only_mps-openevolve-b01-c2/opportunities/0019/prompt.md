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
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249973, "training_seconds": 51.974545875098556, "validation_accuracy": 0.9296, "validation_correct": 9296, "validation_cross_entropy": 0.2071575668334961, "validation_score": 9296.414196136227}
prior_hypothesis: Adding lightweight channel attention to Reference Design 3’s residual features will exceed 9,310 correct validation predictions while remaining below 250,000 learned parameters.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 73.88488316698931, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.20460558624267577, "validation_score": 9330.415073618877}
prior_hypothesis: Annealing from equal ensemble/view supervision to pure ensemble supervision, while preserving the successful 75% average ensemble weight, will exceed 9,325 correct predictions.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.3411985419225, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.20006971549987793, "validation_score": 9322.416642461301}
prior_hypothesis: Restoring the best batch-64 ensemble-loss curriculum and annealing label smoothing only during the final quarter will exceed 9,330 correct predictions by preserving early regularization while better matching hard-label validation late in training.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Replacing arithmetic logit averaging with a calibrated mixture of the two views’ predictive probabilities will exceed 9,312 correct predictions by handling view disagreement without changing the proven architecture or training budget.
change: Use log-probability mixture aggregation for paired-flip supervision and validation inference while retaining per-view deep supervision.
mechanism: Predictive-probability flip ensembling
evidence_used: Paired-flip ensemble training improved accuracy from 9,247 to 9,261, and per-view supervision raised it to 9,276; the current residual model reached 9,312, motivating an isolated refinement of its successful flip-ensemble calculation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 67.150929040974, "validation_accuracy": 0.9297, "validation_correct": 9297, "validation_cross_entropy": 0.20422094345092773, "validation_score": 9297.41520619843}

RECENT RESULT
hypothesis: Adding mild symmetric-KL agreement between paired horizontal views at the proven batch size of 64 will exceed 9,312 correct predictions by reducing view disagreement while preserving the successful logit-averaged objective.
change: Restore the best verified batch size and augment paired-flip deep supervision with a lightweight symmetric-KL consistency penalty.
mechanism: Symmetric flip-view consistency regularization
evidence_used: Batch size 64 produced the best result of 9,312 correct, while adding individual-view supervision previously improved paired-flip training from 9,261 to 9,276; explicit view agreement is the focused remaining extension of that successful mechanism.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Restoring the best batch-64 residual model and weighting its validation-matched ensemble loss at 75% will exceed 9,312 correct predictions while retaining useful per-view supervision.
change: Remove harmful channel attention, restore batch size 64, and change the paired-flip objective from equal weighting to 75% ensemble loss and 25% individual-view loss.
mechanism: Ensemble-dominant paired-flip supervision
evidence_used: The channel-attention design scored 9,296 versus 9,312 for the ungated batch-64 model; paired-view supervision previously improved 9,261 to 9,276, but its equal weighting with the inference-aligned ensemble loss remains untested.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 84.06296679098159, "validation_accuracy": 0.9325, "validation_correct": 9325, "validation_cross_entropy": 0.20271265983581543, "validation_score": 9325.415726895291}

RECENT RESULT
hypothesis: Annealing from equal ensemble/view supervision to pure ensemble supervision, while preserving the successful 75% average ensemble weight, will exceed 9,325 correct predictions.
change: Linearly increase the ensemble-loss weight from 0.5 to 1.0 across training and decrease the per-view weight correspondingly.
mechanism: Validation-aligned flip-loss curriculum
evidence_used: Static 75% ensemble weighting reported 9,325 correct versus 9,312 for equal weighting, while earlier per-view supervision improved 9,261 to 9,276; this schedule retains per-view guidance early and emphasizes the validation-matched ensemble late.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 73.88488316698931, "validation_accuracy": 0.933, "validation_correct": 9330, "validation_cross_entropy": 0.20460558624267577, "validation_score": 9330.415073618877}

RECENT RESULT
hypothesis: Restoring the best batch-64 ensemble-loss curriculum and annealing label smoothing only during the final quarter will exceed 9,330 correct predictions by preserving early regularization while better matching hard-label validation late in training.
change: Use batch size 64, linearly shift supervision from equal view/ensemble loss to pure ensemble loss, and decay label smoothing from 0.02 to zero over the final 25% of steps.
mechanism: Terminal hard-label ensemble fine-tuning
evidence_used: Linear ensemble-weight annealing achieved 9,330 correct versus 9,325 for static 75% weighting and 9,312 for equal weighting, showing that increasingly validation-aligned late supervision is beneficial; terminal smoothing decay extends that mechanism without removing early regularization.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249393, "training_seconds": 74.3411985419225, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.20006971549987793, "validation_score": 9322.416642461301}



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
