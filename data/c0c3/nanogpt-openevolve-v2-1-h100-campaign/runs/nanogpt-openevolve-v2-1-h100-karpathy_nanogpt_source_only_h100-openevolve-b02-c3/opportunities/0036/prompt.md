# Improve fixed-time language-model pretraining

You are an autonomous ML engineer improving the source code for single-GPU
language-model pretraining.

## Goal

Minimize validation bits per byte (`val_bpb`) after a fixed five-minute training
window on the supplied H100 worker. Lower is better. Startup, compilation, and
final validation are outside the measured training window, and every submitted
version starts from a fresh initialization.

You may change the architecture, optimizer, schedules, batching, numerical
implementation, or other contents of `train.py`. The fixed data preparation,
tokenizer, validation procedure, hardware class, and time accounting are not
editable. A useful change must produce a complete trainable implementation and
finish with the required summary metrics.

## Work boundaries

Minimize val_bpb. No additional accuracy threshold.
Editable source files: train.py.
Results reported after each verification: val_bpb, training_seconds, peak_vram_mb, mfu_percent, total_tokens_M, num_steps, num_params_M, depth.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or validation yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"depth": 8.0, "mfu_percent": 37.36, "num_params_M": 50.3, "num_steps": 971.0, "peak_vram_mb": 45067.2, "total_tokens_M": 509.1, "training_seconds": 300.2, "val_bpb": 0.991926}
prior_hypothesis: Using 1.75× MLPs in layers 0–1, 3.25× in layers 2–3, and 5.5× in layers 4–7 will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.

REFERENCE DESIGN 1
verified_results: {"depth": 8.0, "mfu_percent": 37.46, "num_params_M": 50.3, "num_steps": 974.0, "peak_vram_mb": 45068.0, "total_tokens_M": 510.7, "training_seconds": 300.2, "val_bpb": 0.991682}
prior_hypothesis: A 2.25×/2.75× lower-half taper with uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.

REFERENCE DESIGN 2
verified_results: {"depth": 8.0, "mfu_percent": 37.29, "num_params_M": 50.3, "num_steps": 969.0, "peak_vram_mb": 45065.5, "total_tokens_M": 508.0, "training_seconds": 300.1, "val_bpb": 0.991793}
prior_hypothesis: Shifting the lower-half allocation from 2×/3× to 1.5×/3.5× while retaining uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.

REFERENCE DESIGN 3
verified_results: {"depth": 8.0, "mfu_percent": 37.41, "num_params_M": 50.3, "num_steps": 973.0, "peak_vram_mb": 45064.7, "total_tokens_M": 510.1, "training_seconds": 300.3, "val_bpb": 0.991736}
prior_hypothesis: Using 2× MLPs in layers 0–1 and 3× MLPs in layers 2–3, while retaining uniform 5.5× upper-layer MLPs, will preserve roughly 500M-token throughput and reduce val_bpb below 0.991816.

## Recent verification evidence

RECENT RESULT
hypothesis: Preserving the best 2.5× lower-half allocation while shifting upper-half capacity from 5.5× uniformly to 5×/6× by depth will retain at least 495M tokens and reduce val_bpb below 0.991816.
change: Use 2.5× MLPs in layers 0–3, 5× in layers 4–5, and 6× in layers 6–7, preserving the proven 4× average width, optimizer, attention geometry, and schedule.
mechanism: Top-quarter-concentrated ReLU² feed-forward capacity
evidence_used: The 2.5×/5.5× taper achieved the best observed 0.991816 val_bpb at 509.1M tokens, after progressively shifting capacity upward improved every tested taper through that point; redistributing the same upper-half capacity toward the final layers tests whether that depth bias continues without increasing aggregate parameters or MLP compute.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.16, "num_params_M": 50.3, "num_steps": 966.0, "peak_vram_mb": 45066.2, "total_tokens_M": 506.5, "training_seconds": 300.2, "val_bpb": 0.992466}

RECENT RESULT
hypothesis: Using 2× MLPs in layers 0–1 and 3× MLPs in layers 2–3, while retaining uniform 5.5× upper-layer MLPs, will preserve roughly 500M-token throughput and reduce val_bpb below 0.991816.
change: Redistribute the lower half’s unchanged average 2.5× MLP capacity toward its later layers; leave the best-performing upper-half allocation and all other settings unchanged.
mechanism: Lower-half depth-localized feed-forward reallocation
evidence_used: The 2.5×/5.5× design achieved the best val_bpb of 0.991816, while concentrating upper-half capacity into 5×/6× regressed to 0.992466; isolating redistribution within the lower half tests whether finer upward depth bias helps without disturbing the proven upper allocation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.41, "num_params_M": 50.3, "num_steps": 973.0, "peak_vram_mb": 45064.7, "total_tokens_M": 510.1, "training_seconds": 300.3, "val_bpb": 0.991736}

RECENT RESULT
hypothesis: Shifting the lower-half allocation from 2×/3× to 1.5×/3.5× while retaining uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.
change: Use 1.5× MLPs in layers 0–1, 3.5× in layers 2–3, and 5.5× in layers 4–7, preserving total parameters, aggregate MLP compute, and all other settings.
mechanism: Stronger lower-half depth-localized feed-forward reallocation
evidence_used: Redistributing the best 2.5× lower-half capacity to 2×/3× produced the best observed val_bpb of 0.991736 at 510.1M tokens, while redistributing upper-layer capacity regressed; this tests one further lower-only shift without disturbing the proven upper allocation.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.29, "num_params_M": 50.3, "num_steps": 969.0, "peak_vram_mb": 45065.5, "total_tokens_M": 508.0, "training_seconds": 300.1, "val_bpb": 0.991793}

RECENT RESULT
hypothesis: Using 1.75× MLPs in layers 0–1, 3.25× in layers 2–3, and 5.5× in layers 4–7 will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.
change: Restore the proven uniform 5.5× upper-half allocation and interpolate the lower-half taper between the tested 2×/3× and 1.5×/3.5× designs while preserving total parameters and aggregate MLP compute.
mechanism: Midpoint lower-half depth-localized feed-forward reallocation
evidence_used: The 2×/3× lower taper achieved the best observed 0.991736, while strengthening it to 1.5×/3.5× produced a near-tied 0.991793 at comparable throughput; their tensor-core-aligned midpoint most directly refines this narrowly bracketed optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.36, "num_params_M": 50.3, "num_steps": 971.0, "peak_vram_mb": 45067.2, "total_tokens_M": 509.1, "training_seconds": 300.2, "val_bpb": 0.991926}

RECENT RESULT
hypothesis: A 2.25×/2.75× lower-half taper with uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.
change: Use 2.25× MLPs in layers 0–1, 2.75× in layers 2–3, and 5.5× in layers 4–7, preserving total parameters and aggregate MLP compute.
mechanism: Midpoint lower-half feed-forward taper
evidence_used: Uniform 2.5× lower MLPs achieved 0.991816, while shifting to 2×/3× improved to 0.991736 and stronger shifts regressed; this tests the unmeasured midpoint on the weaker-taper side of the apparent optimum.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 37.46, "num_params_M": 50.3, "num_steps": 974.0, "peak_vram_mb": 45068.0, "total_tokens_M": 510.7, "training_seconds": 300.2, "val_bpb": 0.991682}

RECENT RESULT
hypothesis: Shifting upper-half MLP capacity from uniform 5.5× to 6× in layers 4–5 and 5× in layers 6–7 will preserve roughly 500M-token throughput and reduce val_bpb below 0.991682.
change: Retain the best 2.25×/2.75× lower-half taper while redistributing unchanged upper-half capacity toward layers 4–5.
mechanism: Middle-depth-concentrated ReLU² capacity allocation
evidence_used: The current 2.25×/2.75×/5.5× design achieved the best val_bpb of 0.991682, while concentrating upper capacity in the final two layers with a 5×/6× split regressed to 0.992466; testing the reverse 6×/5× split directly probes whether capacity is more useful at middle depth without changing aggregate parameters or MLP compute.
result: the implementation could not be verified

RECENT RESULT
hypothesis: On the best 2.25×/2.75×/5.5× MLP taper, a separate horizon-2 head trained on every fourth position with weight 0.2 will retain at least 490M tokens and reduce val_bpb below 0.991682.
change: Restore the best verified MLP allocation and challenge the assumption that next-token supervision alone produces the best context representation; training adds sparse token-t+2 prediction while validation and primary predictions remain unchanged.
mechanism: Sparse horizon-2 auxiliary prediction
evidence_used: The 2.25×/2.75×/5.5× design achieved 0.991682 at 510.7M tokens, while neighboring width reallocations and SwiGLU did not improve it; preserving ReLU² and introducing a distinct prediction horizon tests a new mechanism with limited compute overhead.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.28, "num_params_M": 54.5, "num_steps": 917.0, "peak_vram_mb": 46229.7, "total_tokens_M": 480.8, "training_seconds": 300.0, "val_bpb": 1.000046}

RECENT RESULT
hypothesis: A 2.125×/2.875× lower-half taper with uniform 5.5× upper MLPs will process roughly 500M tokens and reduce val_bpb below 0.991682.
change: Interpolate between the best 2.25×/2.75× allocation and the competitive 2×/3× allocation while preserving total parameters, aggregate MLP compute, and all other settings.
mechanism: Refined lower-half feed-forward depth taper
evidence_used: The 2.25×/2.75× taper achieved the best val_bpb of 0.991682, narrowly outperforming 2×/3× at 0.991736; their tensor-core-aligned midpoint directly refines the strongest bracket.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.4, "num_params_M": 50.3, "num_steps": 972.0, "peak_vram_mb": 45063.7, "total_tokens_M": 509.6, "training_seconds": 300.1, "val_bpb": 0.991814}

RECENT RESULT
hypothesis: Restoring the verified 2.25×/2.75×/5.5× MLP taper and replacing its linear final-50% warmdown with an equal-duration cosine warmdown will process at least 500M tokens and reduce val_bpb below 0.991682.
change: Restore the best verified MLP allocation, then use cosine rather than linear learning-rate decay during the existing 50% warmdown without changing peak or final learning rates.
mechanism: Cosine terminal annealing on the best depth-tapered ReLU² model
evidence_used: The 2.25×/2.75×/5.5× taper achieved the best val_bpb, 0.991682, while its 2.125×/2.875× refinement regressed to 0.991814 and auxiliary prediction regressed to 1.000046; this motivates preserving the best architecture and testing the previously fixed annealing shape.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.27, "num_params_M": 50.3, "num_steps": 969.0, "peak_vram_mb": 45064.8, "total_tokens_M": 508.0, "training_seconds": 300.2, "val_bpb": 0.99501}

RECENT RESULT
hypothesis: Extending linear warmdown from 50% to 60% on the best 2.25×/2.75×/5.5× architecture will preserve roughly 500M-token throughput and reduce val_bpb below 0.991682.
change: Start the proven linear decay at 40% rather than 50% of the training window while retaining the same peak and zero final learning rates.
mechanism: Earlier linear learning-rate annealing
evidence_used: Equal-duration cosine decay regressed sharply to 0.995010; because cosine maintains a higher learning rate through early warmdown than linear decay, this motivates testing earlier linear annealing without changing the best architecture.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 37.28, "num_params_M": 50.3, "num_steps": 969.0, "peak_vram_mb": 45068.0, "total_tokens_M": 508.0, "training_seconds": 300.1, "val_bpb": 0.992023}

RECENT RESULT
hypothesis: A 2.375×/2.625× lower-half taper with uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991682.
change: Restore the best architecture’s aggregate MLP allocation while testing the tensor-core-aligned midpoint between its 2.25×/2.75× lower taper and the competitive uniform 2.5× lower allocation.
mechanism: Refined weak-side lower-half feed-forward taper
evidence_used: The 2.25×/2.75× taper achieved the best observed 0.991682, outperforming uniform 2.5× at 0.991816; the previously tested stronger-side midpoint regressed, so this tests the unmeasured weaker-side midpoint.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.98, "num_params_M": 50.3, "num_steps": 961.0, "peak_vram_mb": 45067.3, "total_tokens_M": 503.8, "training_seconds": 300.1, "val_bpb": 0.992481}

RECENT RESULT
hypothesis: A 2×/2.5×/2.5×/3× lower-layer MLP ramp with uniform 5.5× upper MLPs will process at least 500M tokens and reduce val_bpb below 0.991682.
change: Preserve the best design’s lower-quarter averages and total MLP capacity while redistributing capacity within each lower-layer pair toward greater depth.
mechanism: Recursive per-layer lower-half feed-forward taper
evidence_used: The 2.25×/2.75× lower-half taper achieved the best observed val_bpb of 0.991682, outperforming both uniform 2.5× and stronger pairwise tapers; recursively applying the same ±0.25× shift within those pairs tests smoother depth localization without changing aggregate parameters or MLP compute.
result: the implementation could not be verified



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
