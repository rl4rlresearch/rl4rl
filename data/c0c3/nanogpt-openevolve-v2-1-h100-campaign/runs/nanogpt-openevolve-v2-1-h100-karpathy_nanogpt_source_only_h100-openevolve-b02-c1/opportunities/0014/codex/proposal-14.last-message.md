MECHANISM: Token-conditioned lexical confidence gate

HYPOTHESIS: A neutral-initialized scalar gate keyed by the current token will reduce val_bpb below 0.994364 by learning when to trust the successful lexical residual, with negligible throughput cost.

INTENDED_EDIT: Add one learned gate per vocabulary token, multiply it into the channel-wise lexical residual, and optimize it with the low-rate scalar parameter group.

EVIDENCE: Channel-wise lexical scaling achieved the best val_bpb of 0.994364, while a dense context-conditioned gate regressed to 0.995708. A scalar token lookup preserves the best model’s initial function while cheaply restoring token-specific lexical confidence removed by normalizing the value embedding.

<<<<<<< SEARCH
        # Per-channel strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        # Value embeddings
=======
        # Per-channel strength and per-token confidence of the direct lexical expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        self.bigram_token_gate = nn.Embedding(config.vocab_size, 1)
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.x0_lambdas.fill_(0.1)
        self.bigram_lambdas.fill_(0.1)
        # Value embeddings
=======
        self.x0_lambdas.fill_(0.1)
        self.bigram_lambdas.fill_(0.1)
        self.bigram_token_gate.weight.zero_()
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel())
=======
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel() + self.bigram_token_gate.weight.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel())
=======
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel() + self.bigram_token_gate.weight.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        resid_params = [self.resid_lambdas, self.bigram_lambdas]
        x0_params = [self.x0_lambdas]
=======
        resid_params = [self.resid_lambdas, self.bigram_lambdas, self.bigram_token_gate.weight]
        x0_params = [self.x0_lambdas]
>>>>>>> REPLACE

<<<<<<< SEARCH
        lexical = norm(ve)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1)
        x = norm(x + lexical)
=======
        lexical = norm(ve)
        token_gate = 2 * torch.sigmoid(self.bigram_token_gate(idx))
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1) * token_gate
        x = norm(x + lexical)
>>>>>>> REPLACE