MECHANISM: Role-specific bigram input embeddings

HYPOTHESIS: Encoding the preceding token directly with a separate role-specific embedding will let all eight blocks operate on bigram-aware representations, lowering val_bpb below 0.992286 while preserving the essential final full-context layer.

INTENDED_EDIT: Add a separate previous-token embedding table, shift it causally, combine it with the current-token embedding before normalization, and include it in initialization, optimization, parameter reporting, and FLOP exclusions.

EVIDENCE: All-local attention processed more tokens but regressed from 0.992286 to 1.015479, showing that throughput and stacked receptive-field reach alone are insufficient; explicitly composing adjacent tokens before attention tests a different context representation without removing global consolidation.

<<<<<<< SEARCH
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
=======
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "prev_wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
=======
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.transformer.prev_wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
=======
        self.transformer.wte.to(dtype=torch.bfloat16)
        self.transformer.prev_wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
>>>>>>> REPLACE

<<<<<<< SEARCH
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
=======
        nparams_exclude = (self.transformer.wte.weight.numel() +
                          self.transformer.prev_wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
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
        prev_wte = sum(p.numel() for p in self.transformer.prev_wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + prev_wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'prev_wte': prev_wte, 'value_embeds': value_embeds,
            'lm_head': lm_head, 'transformer_matrices': transformer_matrices,
            'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_params = list(self.transformer.wte.parameters())
=======
        embedding_params = (list(self.transformer.wte.parameters()) +
                            list(self.transformer.prev_wte.parameters()))
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
=======
        x = self.transformer.wte(idx)
        prev_x = self.transformer.prev_wte(idx)
        prev_x = torch.cat((torch.zeros_like(prev_x[:, :1]), prev_x[:, :-1]), dim=1)
        x = norm(x + prev_x)
        x0 = x
>>>>>>> REPLACE