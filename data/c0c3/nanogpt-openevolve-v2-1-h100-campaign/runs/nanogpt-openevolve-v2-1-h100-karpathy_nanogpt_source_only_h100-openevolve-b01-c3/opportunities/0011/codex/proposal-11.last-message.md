MECHANISM: Higher Muon step size with full polar refinement

HYPOTHESIS: Restoring five-step Muon orthogonalization and increasing matrix LR by 25% will make each of the proven 948 updates more effective, reducing val_bpb below 0.995558 without changing throughput.

INTENDED_EDIT: Restore five Muon polar iterations and raise MATRIX_LR from 0.04 to 0.05 while preserving the successful SSSL architecture and 524,288-token batch.

EVIDENCE: Five iterations achieved 0.995558 versus 0.998073 for three iterations at identical 497.0M tokens and 948 steps; the 786,432-token batch’s 632 updates produced 1.012938, motivating stronger per-update progress at the successful batch size.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE

<<<<<<< SEARCH
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
=======
MATRIX_LR = 0.05        # stronger matrix updates with full five-step Muon refinement
>>>>>>> REPLACE