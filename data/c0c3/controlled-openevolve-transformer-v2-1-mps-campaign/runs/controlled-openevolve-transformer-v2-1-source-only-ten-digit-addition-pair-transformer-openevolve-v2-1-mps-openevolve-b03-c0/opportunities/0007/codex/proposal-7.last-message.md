MECHANISM: Shared normalization offset across block and output norms

HYPOTHESIS: Tying `ln_f.bias` to the already shared block LayerNorm bias will reduce deduplicated parameters from 1,628 to 1,620 while retaining at least 99% accuracy.

INTENDED_EDIT: Share the final LayerNorm bias with `blocks[0].ln1.bias`, preserving separate normalization scales and all attention and MLP capacity.

EVIDENCE: Sharing both block LayerNorm affine parameters achieved 99.59% accuracy at 1,628 parameters; this motivates an isolated eight-parameter bias-sharing extension rather than reducing the capacity components that previously failed.

<<<<<<< SEARCH
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.ln_f.bias = self.blocks[0].ln1.bias

        # Weight tying with input embeddings.
>>>>>>> REPLACE