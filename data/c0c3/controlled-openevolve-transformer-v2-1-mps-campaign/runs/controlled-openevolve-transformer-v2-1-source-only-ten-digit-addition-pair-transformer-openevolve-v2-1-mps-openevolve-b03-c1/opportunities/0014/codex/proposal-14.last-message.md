MECHANISM: Single-coordinate token–position embedding gauge fixing

HYPOTHESIS: Fixing one positional-embedding coordinate at zero through an initialization-aligned token/position reparameterization will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Absorb position zero’s first coordinate into the corresponding coordinate of every token embedding, compact the positional parameter by one scalar, and reconstruct the fixed zero coordinate during the forward pass.

EVIDENCE: The current 1,604-parameter model achieved 99.37% accuracy, and RNG-aligned removal of an exact attention redundancy previously succeeded where a non-aligned version failed; this patch likewise preserves the initialized model function exactly while testing a different one-dimensional gauge after the MLP-bias gauge reached only 98.24%.

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)

        # Fix one exact token/position embedding gauge while preserving the
        # initialized function: input embeddings are unchanged, and the tied
        # output weights gain only a token-independent logit shift.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, 0].clone()
            self.token_emb.weight[:, 0].add_(offset)
            shifted_pos = self.pos_emb.weight.clone()
            shifted_pos[:, 0].sub_(offset)
            compact_pos = torch.cat((shifted_pos[0, 1:], shifted_pos[1:].reshape(-1)))
        self.pos_emb.weight = nn.Parameter(compact_pos)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        pos_weight = torch.cat(
            (self.pos_emb.weight.new_zeros(1), self.pos_emb.weight)
        ).view(self.cfg.max_seq_len, self.cfg.d_model)
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
>>>>>>> REPLACE