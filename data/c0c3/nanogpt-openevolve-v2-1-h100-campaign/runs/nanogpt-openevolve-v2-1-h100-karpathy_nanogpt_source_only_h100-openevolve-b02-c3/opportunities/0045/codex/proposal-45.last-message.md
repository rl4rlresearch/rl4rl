MECHANISM: Token-normalized optimizer-batch midpoint

HYPOTHESIS: A 212,992-token batch with learning rates scaled by 13/14 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving cumulative optimizer displacement while interpolating between the two equally competitive normalized-batch endpoints.

INTENDED_EDIT: Use a 104-sequence single microbatch and scale every optimizer-group peak learning rate by the batch ratio 212,992/229,376.

EVIDENCE: The 229,376-token baseline reached 0.984024, while 196,608 tokens with proportional 6/7 learning rates reached 0.984034; their near-identical results bracket this untested midpoint and support keeping learning rate proportional to batch size.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 7 * 2**15 # ~229K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 13 * 2**14 # ~213K tokens per optimizer step
EMBEDDING_LR = 0.6 * 13 / 14      # batch-normalized token embedding LR
UNEMBEDDING_LR = 0.004 * 13 / 14  # batch-normalized lm_head LR
MATRIX_LR = 0.04 * 13 / 14        # batch-normalized Muon matrix LR
SCALAR_LR = 0.5 * 13 / 14         # batch-normalized scalar LR
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 112  # one 229K-token microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 104  # one 213K-token microbatch per optimizer step
>>>>>>> REPLACE