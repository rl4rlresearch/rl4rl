MECHANISM: Intermediate-batch update densification

HYPOTHESIS: A 196,608-token batch will retain at least 450M-token throughput while providing roughly 2,350 updates, reducing val_bpb below the 262,144-token batch’s 0.985506.

INTENDED_EDIT: Remove the unsuccessful hashed-bigram path, restore the proven uniform 4.375× MLP model, and use a single-microbatch 196,608-token global batch.

EVIDENCE: Reducing batch size from 524,288 to 262,144 improved val_bpb from 0.994296 to 0.985506, but reducing again to 131,072 worsened it to 0.989237; the untested midpoint directly probes the apparent optimum.

<<<<<<< SEARCH
        # A pair of coprime hash tables gives each (previous, current) token pair
        # a nearly unique compositional code without a vocabulary-squared table.
        assert config.n_embd % 2 == 0
        self.bigram_bucket_sizes = (4093, 4099)
        self.bigram_embeds = nn.ModuleList([
            nn.Embedding(bucket_size, config.n_embd // 2)
            for bucket_size in self.bigram_bucket_sizes
        ])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        for bigram_embed in self.bigram_embeds:
            torch.nn.init.normal_(bigram_embed.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
=======
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.transformer.wte.to(dtype=torch.bfloat16)
        for bigram_embed in self.bigram_embeds:
            bigram_embed.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
=======
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        bigram_embeds_numel = sum(embed.weight.numel() for embed in self.bigram_embeds)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          bigram_embeds_numel + self.resid_lambdas.numel() +
                          self.x0_lambdas.numel())
=======
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_params = (list(self.transformer.wte.parameters()) +
                            list(self.bigram_embeds.parameters()))
=======
        embedding_params = list(self.transformer.wte.parameters())
>>>>>>> REPLACE

<<<<<<< SEARCH
        prev_idx = F.pad(idx[:, :-1], (1, 0), value=0)
        pair_idx = prev_idx * self.config.vocab_size + idx
        bigram = torch.cat((
            self.bigram_embeds[0](pair_idx.remainder(self.bigram_bucket_sizes[0])),
            self.bigram_embeds[1](pair_idx.remainder(self.bigram_bucket_sizes[1])),
        ), dim=-1)

        x = self.transformer.wte(idx) + 0.25 * bigram
=======
        x = self.transformer.wte(idx)
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
>>>>>>> REPLACE