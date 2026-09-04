MECHANISM: Residual low-rank bigram logit expert

HYPOTHESIS: A zero-initialized rank-64 bigram expert at the proven 93.75% learning rates will retain at least 450M-token throughput and reduce val_bpb below 0.984614 by learning frequent token transitions outside the deep contextual path.

INTENDED_EDIT: Challenge the assumption that every prediction must be decoded solely from the final transformer state; add an exact, collision-free factorized bigram distribution directly to the logits while retaining the transformer for longer-context corrections.

EVIDENCE: The 93.75% learning-rate design is best at 0.984614. The hashed-bigram design reached only 0.994892 despite 466.6M tokens, indicating that indirect, collision-prone residual injection was ineffective; a zero-initialized direct logit expert tests the local-statistics idea without requiring eight layers to preserve and decode the added representation.

<<<<<<< SEARCH
    n_embd: int = 768
    window_pattern: str = "SSSL"
=======
    n_embd: int = 768
    window_pattern: str = "SSSL"
    bigram_rank: int = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Low-rank transition expert: logits[next_token | current_token].
        self.bigram_embed = nn.Embedding(config.vocab_size, config.bigram_rank)
        self.bigram_head = nn.Linear(config.bigram_rank, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
=======
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.bigram_embed.weight, mean=0.0, std=1.0)
        # Preserve the baseline function initially; the expert learns a residual.
        torch.nn.init.zeros_(self.bigram_head.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
=======
        self.transformer.wte.to(dtype=torch.bfloat16)
        self.bigram_embed.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
=======
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.bigram_embed.weight.numel() +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
=======
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        bigram = sum(p.numel() for p in self.bigram_embed.parameters())
        bigram += sum(p.numel() for p in self.bigram_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + bigram + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'bigram': bigram, 'transformer_matrices': transformer_matrices,
            'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
=======
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        bigram_embed_params = list(self.bigram_embed.parameters())
        bigram_head_params = list(self.bigram_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(bigram_embed_params) + len(bigram_head_params) +
            len(value_embeds_params) + len(resid_params) + len(x0_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=lm_head_params + bigram_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params + bigram_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
        logits = logits.float()
=======
        logits = self.lm_head(x)
        bigram_logits = self.bigram_head(norm(self.bigram_embed(idx)))
        logits = (logits + bigram_logits).float()
>>>>>>> REPLACE

<<<<<<< SEARCH
EMBEDDING_LR = 0.525    # 87.5% of the proven embedding LR
UNEMBEDDING_LR = 0.0035 # 87.5% of the proven lm_head LR
MATRIX_LR = 0.035       # 87.5% of the proven Muon LR
SCALAR_LR = 0.4375      # 87.5% of the proven per-layer scalar LR
=======
EMBEDDING_LR = 0.5625   # 93.75% of the proven embedding LR
UNEMBEDDING_LR = 0.00375 # 93.75% of the proven lm_head LR
MATRIX_LR = 0.0375      # 93.75% of the proven Muon LR
SCALAR_LR = 0.46875     # 93.75% of the proven per-layer scalar LR
>>>>>>> REPLACE