MECHANISM: Third-coordinate token–position embedding gauge fixing

HYPOTHESIS: Extending the successful positional-embedding gauge fixing from two coordinates to three will reduce parameters from 1,602 to 1,601 while retaining at least 99% accuracy.

INTENDED_EDIT: Absorb position zero’s first three coordinates into every token embedding, remove those three positional scalars, and reconstruct them as zeros during the forward pass.

EVIDENCE: The initialization-aligned one- and two-coordinate gauge fixes achieved 99.92% and 99.75% accuracy respectively, directly motivating one more incremental removal of the same exact redundancy.

<<<<<<< SEARCH
        # Fix two exact token/position embedding gauges while preserving the
        # initialized function: input embeddings are unchanged, and the tied
        # output weights gain only token-independent logit shifts.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, :2].clone()
            self.token_emb.weight[:, :2].add_(offset)
            shifted_pos = self.pos_emb.weight.clone()
            shifted_pos[:, :2].sub_(offset)
            compact_pos = torch.cat((shifted_pos[0, 2:], shifted_pos[1:].reshape(-1)))
=======
        # Fix three exact token/position embedding gauges while preserving the
        # initialized function: input embeddings are unchanged, and the tied
        # output weights gain only token-independent logit shifts.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, :3].clone()
            self.token_emb.weight[:, :3].add_(offset)
            shifted_pos = self.pos_emb.weight.clone()
            shifted_pos[:, :3].sub_(offset)
            compact_pos = torch.cat((shifted_pos[0, 3:], shifted_pos[1:].reshape(-1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (self.pos_emb.weight.new_zeros(2), self.pos_emb.weight)
        ).view(self.cfg.max_seq_len, self.cfg.d_model)
=======
        pos_weight = torch.cat(
            (self.pos_emb.weight.new_zeros(3), self.pos_emb.weight)
        ).view(self.cfg.max_seq_len, self.cfg.d_model)
>>>>>>> REPLACE