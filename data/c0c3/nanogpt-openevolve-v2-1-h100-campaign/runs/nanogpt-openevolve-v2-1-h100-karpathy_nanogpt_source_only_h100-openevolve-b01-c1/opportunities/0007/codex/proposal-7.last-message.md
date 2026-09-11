MECHANISM: Two-to-one grouped-query attention

HYPOTHESIS: Halving KV heads while retaining four query heads and the final full-context layer will reduce parameter-update and projection cost enough to process more tokens and lower val_bpb below 0.992286.

INTENDED_EDIT: Configure two KV heads for the four-head model, preserving every attention window and query head.

EVIDENCE: Removing one full-context layer increased training from 522.2M to 530.6M tokens and improved val_bpb from 0.993215 to 0.992286, while removing the final global layer sharply regressed; this motivates reducing attention cost without sacrificing global integration.

<<<<<<< SEARCH
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
=======
    num_kv_heads = max(1, num_heads // 2)
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
>>>>>>> REPLACE