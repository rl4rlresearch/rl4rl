MECHANISM: Anchored relative positional embeddings

HYPOTHESIS: Fixing position 0’s seven positional coordinates to zero will reduce the model from 1,280 to 1,273 parameters while retaining at least 99% accuracy, because any removed position-origin vector can be absorbed uniformly into the tied token embeddings, preserving inputs while changing output logits only by a class-common offset.

INTENDED_EDIT: Store learned seven-coordinate embeddings only for positions 1 through `max_seq_len - 1` and prepend a fixed-zero embedding for position 0 during the forward pass.

EVIDENCE: Gauge-fixing each positional embedding’s unobservable common-mode coordinate retained 100% accuracy at 1,280 parameters; anchoring the shared positional origin removes another exact positional/token-embedding translation redundancy without restricting relative positional freedom.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + F.pad(self.pos_emb(pos), (0, 1))
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = self.token_emb(idx) + F.pad(F.embedding(pos, pos_weight), (0, 1))
        x = self.drop(x)
>>>>>>> REPLACE