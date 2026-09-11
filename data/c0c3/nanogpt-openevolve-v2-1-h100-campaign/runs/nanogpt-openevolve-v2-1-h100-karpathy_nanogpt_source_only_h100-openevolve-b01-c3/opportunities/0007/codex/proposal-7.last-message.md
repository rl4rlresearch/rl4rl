MECHANISM: Grouped-query KV compression with restored full-context layers

HYPOTHESIS: Halving the KV-head count while restoring SSSL will reduce projection and value-embedding cost enough to process more than 497M tokens and achieve val_bpb below 0.995558 without sacrificing query-head capacity.

INTENDED_EDIT: Restore the best-performing SSSL attention pattern and use two KV heads shared across four query heads.

EVIDENCE: SSSS achieved essentially unchanged throughput (496.5M versus 497.0M tokens) while worsening val_bpb to 0.997009, so reducing attention context is ineffective; compressing K/V compute while preserving the successful context allocation is the clearest untested efficiency change.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # half context throughout, with the final layer forced to full context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
>>>>>>> REPLACE

<<<<<<< SEARCH
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
=======
    num_heads = model_dim // HEAD_DIM
    num_kv_heads = max(1, num_heads // 2)
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
>>>>>>> REPLACE