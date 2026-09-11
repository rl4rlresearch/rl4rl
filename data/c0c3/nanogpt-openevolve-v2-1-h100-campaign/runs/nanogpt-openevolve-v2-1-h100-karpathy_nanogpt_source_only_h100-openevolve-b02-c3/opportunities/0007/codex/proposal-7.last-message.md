MECHANISM: Truncated Polar-Express Muon orthogonalization

HYPOTHESIS: Reducing Muon from five to three polar iterations will preserve useful gradient orthogonalization while lowering optimizer overhead, processing more than 513.3M tokens and reducing val_bpb below 0.993287.

INTENDED_EDIT: Retain the proven 512-token SSSL architecture and 524K-token batch, but use three Newton-Schulz-style Polar Express iterations per Muon update.

EVIDENCE: The best design reached 0.993287 val_bpb at 513.3M tokens, while halving the batch reduced throughput to 383.5M tokens despite more updates; this implicates per-update overhead and motivates accelerating the optimizer without changing the proven model geometry.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE