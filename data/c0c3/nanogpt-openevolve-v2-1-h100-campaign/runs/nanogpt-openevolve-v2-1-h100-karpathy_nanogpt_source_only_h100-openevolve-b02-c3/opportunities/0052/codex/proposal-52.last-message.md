MECHANISM: Locally interpolated peak learning rate

HYPOTHESIS: A 212,992-token batch with all peak learning rates scaled by 11/12 will process at least 490M tokens and reduce val_bpb below 0.983505.

INTENDED_EDIT: Preserve the best architecture, batch, and linear schedule while moving the coupled peak-rate scale from 33/35 to 11/12.

EVIDENCE: At fixed batch size, scales 0.90, 0.92, and 0.9286 yielded val_bpb 0.983937, 0.983505, and 0.983699; local quadratic interpolation estimates the minimum near 0.917.

<<<<<<< SEARCH
EMBEDDING_LR = 0.6 * 33 / 35      # upward-refined embedding peak LR
UNEMBEDDING_LR = 0.004 * 33 / 35  # upward-refined lm_head peak LR
MATRIX_LR = 0.04 * 33 / 35        # upward-refined Muon matrix peak LR
SCALAR_LR = 0.5 * 33 / 35         # upward-refined scalar peak LR
=======
EMBEDDING_LR = 0.6 * 11 / 12      # locally interpolated embedding peak LR
UNEMBEDDING_LR = 0.004 * 11 / 12  # locally interpolated lm_head peak LR
MATRIX_LR = 0.04 * 11 / 12        # locally interpolated Muon matrix peak LR
SCALAR_LR = 0.5 * 11 / 12         # locally interpolated scalar peak LR
>>>>>>> REPLACE