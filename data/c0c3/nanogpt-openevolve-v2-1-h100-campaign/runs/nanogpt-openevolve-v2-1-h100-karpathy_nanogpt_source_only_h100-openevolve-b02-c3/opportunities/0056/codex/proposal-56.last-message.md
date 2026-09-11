MECHANISM: Optimizer-specific peak learning rates

HYPOTHESIS: Using the batch-normalized 13/14 peak rate for AdamW parameters while retaining the best 23/25 Muon rate will process at least 490M tokens and reduce val_bpb below 0.983505.

INTENDED_EDIT: Restore the best Muon matrix rate while independently raising embedding, unembedding, and scalar AdamW rates to 13/14 of their base values.

EVIDENCE: The coupled 23/25 design achieved 0.983505, while the coupled 13/14 design was close at 0.983699; decoupling them tests whether the former’s advantage came from Muon matrices while AdamW-managed parameters benefit from the batch-normalized rate.

<<<<<<< SEARCH
EMBEDDING_LR = 0.6 * 33 / 35      # upward-refined embedding peak LR
UNEMBEDDING_LR = 0.004 * 33 / 35  # upward-refined lm_head peak LR
MATRIX_LR = 0.04 * 33 / 35        # upward-refined Muon matrix peak LR
SCALAR_LR = 0.5 * 33 / 35         # upward-refined scalar peak LR
=======
EMBEDDING_LR = 0.6 * 13 / 14      # batch-normalized AdamW embedding peak LR
UNEMBEDDING_LR = 0.004 * 13 / 14  # batch-normalized AdamW lm_head peak LR
MATRIX_LR = 0.04 * 23 / 25        # best verified Muon matrix peak LR
SCALAR_LR = 0.5 * 13 / 14         # batch-normalized AdamW scalar peak LR
>>>>>>> REPLACE