MECHANISM: Lower-side Muon learning-rate midpoint refinement

HYPOTHESIS: Using `MATRIX_LR=0.039` with the proven 258,048-token batch will retain roughly 520M training tokens and lower val_bpb below 0.985181.

INTENDED_EDIT: Reduce only the Muon matrix learning rate from 0.040 to 0.039, preserving the best architecture, batch size, optimizer, and 60% warmdown.

EVIDENCE: At the best batch size, 0.040 achieved 0.985181; equal-sized deviations to 0.038 and 0.042 regressed to 0.985701 and 0.985838 respectively, so the less harmful lower side motivates testing its midpoint.

<<<<<<< SEARCH
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
=======
MATRIX_LR = 0.039       # lower-side midpoint around the best Muon learning rate
>>>>>>> REPLACE