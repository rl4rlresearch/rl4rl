MECHANISM: Fine-grained MLP capacity interpolation

HYPOTHESIS: A 5.125× MLP at the proven 93.75% learning rates will retain at least 445M-token throughput and reduce val_bpb below 0.982905.

INTENDED_EDIT: Remove the unsuccessful output bias and widen every MLP from 4.375× to 5.125× while retaining the proven batch, schedule, and optimizer rates.

EVIDENCE: The unbiased 5× MLP achieved 0.982905, while 5.25× reached 0.983115 and 5.625× reached 0.983497; testing the midpoint between 5× and 5.25× refines the bracket around the observed capacity optimum.

<<<<<<< SEARCH
        hidden_dim = 35 * config.n_embd // 8
=======
        hidden_dim = 41 * config.n_embd // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.lm_head.bias)
=======
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
>>>>>>> REPLACE