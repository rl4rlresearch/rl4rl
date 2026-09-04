MECHANISM: Positional-embedding shift-gauge elimination

HYPOTHESIS: Removing one positional-embedding coordinate via an initialization-preserving per-position constant shift will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because pre-LayerNorm attention and MLP inputs are invariant to that shift and the final LayerNorm removes it exactly.

INTENDED_EDIT: Canonicalize position zero’s embedding so its first coordinate is fixed at zero, store only the remaining positional parameters, and reconstruct the full embedding tensor during the learned forward pass.

EVIDENCE: The current one-key/one-value-bias design achieved 99.96% at 1,618 parameters, while other one-parameter reductions failed despite preserving capacity; this motivates testing a distinct exact model symmetry with both the initialized function and all attention/MLP capacity preserved.

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Remove one unobservable per-position constant-shift degree of freedom.
        pos_weight = self.pos_emb.weight.detach().clone()
        gauge = pos_weight[0, 0].clone()
        pos_weight[0] = pos_weight[0] - gauge
        self.pos_emb.weight = nn.Parameter(pos_weight.reshape(-1)[1:].clone())

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = torch.cat(
            (self.pos_emb.weight.new_zeros(1), self.pos_emb.weight)
        ).view(self.cfg.max_seq_len, self.cfg.d_model)
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
        x = self.drop(x)
>>>>>>> REPLACE