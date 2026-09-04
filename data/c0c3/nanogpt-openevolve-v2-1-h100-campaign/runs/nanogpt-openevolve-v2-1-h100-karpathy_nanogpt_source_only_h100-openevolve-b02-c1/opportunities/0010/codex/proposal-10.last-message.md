MECHANISM: Per-head lexical-residual readout

HYPOTHESIS: Adding a learned low-rank bigram expert from the final value embedding directly to the pre-logit state will reduce val_bpb below 0.995558 without materially reducing throughput.

INTENDED_EDIT: Challenge the assumption that prediction should depend solely on the fully transformed contextual stream; preserve the baseline transformer and blend its final token-value embedding into the readout using four learned per-head coefficients.

EVIDENCE: Grouped-query attention increased training volume to 511.7M tokens but worsened val_bpb to 0.999976, suggesting value-path expressivity is more important than another small throughput gain. This reuses the full-capacity value representation while leaving the successful attention topology unchanged.

<<<<<<< SEARCH
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
=======
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Per-head strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_head,), 0.1))
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
=======
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.bigram_lambdas.fill_(0.1)
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
=======
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
=======
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        resid_params = [self.resid_lambdas]
=======
        resid_params = [self.resid_lambdas, self.bigram_lambdas]
>>>>>>> REPLACE

<<<<<<< SEARCH
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
=======
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])

        # The final layer always has a value embedding. Route it directly to the
        # readout as a per-head low-rank bigram expert alongside contextual state.
        lexical = norm(ve).view(B, T, self.config.n_head, -1)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1, 1)
        x = norm(x + lexical.reshape(B, T, -1))

        softcap = 15
>>>>>>> REPLACE