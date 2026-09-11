MECHANISM: Dual-hash bigram memory in the residual stream

HYPOTHESIS: Replacing the assumption that attention must reconstruct all local token interactions with a low-FLOP learned bigram representation, while restoring the proven 4.375× MLP, will process at least 440M tokens and reduce val_bpb below 0.994296.

INTENDED_EDIT: Restore 2240-channel MLPs and augment each token embedding with a scaled, trainable bigram code formed by concatenating two independently hashed half-width embeddings.

EVIDENCE: The 4.375× MLP achieved 0.994296 at 472.9M tokens, whereas 3.75× achieved 0.996902 despite 498.6M tokens, showing that representational capacity outweighed marginal throughput; hashed lookup capacity adds an explicit local-context mechanism without another dense matrix.

<<<<<<< SEARCH
        hidden_dim = 15 * config.n_embd // 4
=======
        hidden_dim = 35 * config.n_embd // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
=======
        # A pair of coprime hash tables gives each (previous, current) token pair
        # a nearly unique compositional code without a vocabulary-squared table.
        assert config.n_embd % 2 == 0
        self.bigram_bucket_sizes = (4093, 4099)
        self.bigram_embeds = nn.ModuleList([
            nn.Embedding(bucket_size, config.n_embd // 2)
            for bucket_size in self.bigram_bucket_sizes
        ])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
=======
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        for bigram_embed in self.bigram_embeds:
            torch.nn.init.normal_(bigram_embed.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
=======
        self.transformer.wte.to(dtype=torch.bfloat16)
        for bigram_embed in self.bigram_embeds:
            bigram_embed.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
>>>>>>> REPLACE

<<<<<<< SEARCH
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
=======
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        bigram_embeds_numel = sum(embed.weight.numel() for embed in self.bigram_embeds)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          bigram_embeds_numel + self.resid_lambdas.numel() +
                          self.x0_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
=======
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        bigram_embeds = sum(p.numel() for p in self.bigram_embeds.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + bigram_embeds + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'bigram_embeds': bigram_embeds,
            'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_params = list(self.transformer.wte.parameters())
=======
        embedding_params = (list(self.transformer.wte.parameters()) +
                            list(self.bigram_embeds.parameters()))
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
=======
        prev_idx = F.pad(idx[:, :-1], (1, 0), value=0)
        pair_idx = prev_idx * self.config.vocab_size + idx
        bigram = torch.cat((
            self.bigram_embeds[0](pair_idx.remainder(self.bigram_bucket_sizes[0])),
            self.bigram_embeds[1](pair_idx.remainder(self.bigram_bucket_sizes[1])),
        ), dim=-1)

        x = self.transformer.wte(idx) + 0.25 * bigram
        x = norm(x)
        x0 = x
>>>>>>> REPLACE