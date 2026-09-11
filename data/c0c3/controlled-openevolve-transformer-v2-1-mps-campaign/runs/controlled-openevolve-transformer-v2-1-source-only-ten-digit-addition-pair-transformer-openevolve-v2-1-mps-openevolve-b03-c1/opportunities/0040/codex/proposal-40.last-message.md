MECHANISM: Rank-six tied lexical bottleneck

HYPOTHESIS: Replacing the dense 114-by-8 tied token table with a learned rank-six factorization will reduce parameters from 1,439 to 1,259 while retaining at least 99% accuracy, because token identity and output classification can share a six-dimensional learned latent space while the full-width attention pathway remains intact.

INTENDED_EDIT: Initialize the existing dense tied token table, truncate it by SVD into learned token codes and a learned 6-by-8 projection, and use their product for both input embeddings and output logits.

EVIDENCE: The rank-four positional representation achieved 99.96%, showing that useful task representations tolerate substantial latent bottlenecks; conversely, shared-key attention collapsed to 36.45%, motivating compression of the much larger lexical table while preserving all query, key, and value routing capacity.

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Initialize a dense tied lexical table as before, then compress it
        # below into a shared learned latent representation.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()

        # Preserve the baseline RNG stream by initializing the original table.
=======
        for block in self.blocks:
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()

        # Challenge the dense-lexicon assumption shared by the prior designs.
        # A learned rank-six table supplies token-specific latent codes, while
        # a shared learned projection maps those codes into the residual stream.
        # Reusing both factors for the output preserves weight tying.
        token_rank = 6
        with torch.no_grad():
            token_left, token_singular, token_right = torch.linalg.svd(
                self.token_emb.weight, full_matrices=False
            )
            token_scale = token_singular[:token_rank].sqrt()
            token_code = token_left[:, :token_rank] * token_scale
            token_proj = token_scale.unsqueeze(1) * token_right[:token_rank]
        self.token_code = nn.Parameter(token_code.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
        self.token_emb = None
        self.lm_head = None

        # Preserve the baseline RNG stream by initializing the original table.
>>>>>>> REPLACE

<<<<<<< SEARCH
        position = F.embedding(pos, self.pos_code) @ pos_proj
        x = self.token_emb(idx) + position
        x = self.drop(x)
=======
        position = F.embedding(pos, self.pos_code) @ pos_proj
        token = F.embedding(idx, self.token_code) @ self.token_proj
        x = token + position
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        token_features = x @ self.token_proj.T
        logits = F.linear(token_features, self.token_code)
>>>>>>> REPLACE