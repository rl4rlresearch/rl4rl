MECHANISM: Lower-overhead Muon orthogonalization

HYPOTHESIS: Reducing Muon’s polar iterations from five to three will preserve useful update geometry while increasing token throughput enough to beat the current 0.995558 val_bpb.

INTENDED_EDIT: Use three Newton–Schulz/polar-express iterations for every Muon parameter group.

EVIDENCE: Halving the batch doubled update frequency but reduced throughput from 497.0M to 393.2M tokens and worsened val_bpb to 0.998629, indicating per-step optimizer overhead is substantial and favoring faster updates at the successful 524,288-token batch.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE