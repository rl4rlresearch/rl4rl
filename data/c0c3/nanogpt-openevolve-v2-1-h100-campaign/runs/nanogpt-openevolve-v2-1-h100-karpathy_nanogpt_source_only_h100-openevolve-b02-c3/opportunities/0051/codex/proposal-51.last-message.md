MECHANISM: Two-way grouped-query attention

HYPOTHESIS: Using two KV heads for four query heads will process at least 500M tokens while retaining sufficient contextual capacity to reduce val_bpb below 0.983505.

INTENDED_EDIT: Restore the best 23/25 learning-rate scale and replace full multi-head K/V projections with two-group GQA, halving K/V and value-embedding width while preserving query heads, model width, batch, and schedule.

EVIDENCE: The 212,992-token 23/25-rate design achieved the best val_bpb, 0.983505, while doubling full-width heads regressed to 0.989288; this motivates testing whether K/V head redundancy can be removed for fixed-time efficiency rather than adding query-head specialization.

<<<<<<< SEARCH
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
=======
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads // 2, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
EMBEDDING_LR = 0.6 * 9 / 10      # downward-refined embedding peak LR
UNEMBEDDING_LR = 0.004 * 9 / 10  # downward-refined lm_head peak LR
MATRIX_LR = 0.04 * 9 / 10        # downward-refined Muon matrix peak LR
SCALAR_LR = 0.5 * 9 / 10         # downward-refined scalar peak LR
=======
EMBEDDING_LR = 0.6 * 23 / 25      # best verified embedding peak LR
UNEMBEDDING_LR = 0.004 * 23 / 25  # best verified lm_head peak LR
MATRIX_LR = 0.04 * 23 / 25        # best verified Muon matrix peak LR
SCALAR_LR = 0.5 * 23 / 25         # best verified scalar peak LR
>>>>>>> REPLACE