MECHANISM: Near-free unigram logit prior

HYPOTHESIS: A zero-initialized output bias at the proven 93.75% learning rates will retain at least 460M-token throughput and reduce val_bpb below 0.984614 by learning global next-token frequencies without the rank-64 bigram expert’s compute and memory overhead.

INTENDED_EDIT: Restore the best 93.75% optimizer rates and add a zero-initialized bias to the language-model head.

EVIDENCE: The rank-64 direct-logit bigram expert lost 9.4M tokens and worsened val_bpb from 0.984614 to 0.985446; a fused output bias tests whether a minimal direct statistical expert provides the useful prior without that throughput penalty.

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
=======
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.lm_head.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
EMBEDDING_LR = 0.5625   # 93.75% of the proven embedding LR
UNEMBEDDING_LR = 0.00375 # 93.75% of the proven lm_head LR
MATRIX_LR = 0.0375      # 93.75% of the proven Muon LR
SCALAR_LR = 0.46875     # 93.75% of the proven per-layer scalar LR
>>>>>>> REPLACE