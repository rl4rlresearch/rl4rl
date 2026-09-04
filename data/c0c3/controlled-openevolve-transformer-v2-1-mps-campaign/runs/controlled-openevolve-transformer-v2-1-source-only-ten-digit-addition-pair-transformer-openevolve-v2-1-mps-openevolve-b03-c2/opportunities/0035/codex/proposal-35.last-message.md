MECHANISM: Trajectory-preserving LayerNorm-scale folding into attention QKV weights

HYPOTHESIS: Folding all eight `ln1` scales into `qkv.weight` while reproducing their AdamW and clipping dynamics will reduce the verified 1579-parameter model to 1571 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Make `ln1` non-affine and add each attention QKV weight to the optimizer’s existing folded-factor path alongside `fc1.weight`.

EVIDENCE: Folding all eight `ln2` scales into `fc1.weight` with factor-aware AdamW and clipping achieved 99.75% accuracy at 1579 parameters; `ln1` has the same columnwise multiplicative redundancy with its downstream QKV projection.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = None
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        # Its learned scale is folded into fc1.weight by the optimizer.
=======
        # Its learned scale is folded into qkv.weight by the optimizer.
        self.ln1 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        # Its learned scale is folded into fc1.weight by the optimizer.
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Reproduce AdamW on the full fc1 weight and the omitted ln2 scale,
        # then store their functionally sufficient columnwise product.
=======
        # Reproduce AdamW on each downstream weight and its omitted
        # LayerNorm scale, then store their sufficient columnwise product.
>>>>>>> REPLACE

<<<<<<< SEARCH
    factor_params = [
        block.mlp.fc1.weight for block in model.blocks
    ]
=======
    factor_params = [
        block.attn.qkv.weight for block in model.blocks
    ] + [
        block.mlp.fc1.weight for block in model.blocks
    ]
>>>>>>> REPLACE