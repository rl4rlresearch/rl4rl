MECHANISM: Gated low-rank bigram prediction bypass

HYPOTHESIS: Adding a learned direct path from the current-token embedding to the output logits will factorize common token transitions away from the contextual stack, allowing the preserved full-context model and 262K-token update cadence to lower val_bpb below 0.986167 without meaningful throughput loss.

INTENDED_EDIT: Add a zero-initialized per-channel output gate that mixes the normalized input embedding into the final representation before the shared language-model head, with a dedicated conservative AdamW learning rate and complete parameter/FLOP accounting.

EVIDENCE: The 262K-token batch improved val_bpb to 0.986167, while removing full-context integration regressed to 1.015479. The old design assumes one final representation must encode both immediate token-transition statistics and global context; this patch preserves global attention but gives prediction a separate learned low-rank bigram route.

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Per-channel direct token-to-next-token prediction path
        self.output_x0_gate = nn.Parameter(torch.zeros(config.n_embd))
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
=======
        # Per-layer scalars and prediction bypass
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.output_x0_gate.zero_()
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
=======
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.output_x0_gate.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
=======
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.output_x0_gate.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
=======
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        prediction_params = [self.output_x0_gate]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(x0_params) + len(prediction_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=prediction_params, lr=scalar_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
=======
        x = norm(x)
        x = x + self.output_x0_gate * x0

        softcap = 15
        logits = self.lm_head(x)
>>>>>>> REPLACE