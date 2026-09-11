MECHANISM: Downstream-absorbed terminal LayerNorm scale gauge

HYPOTHESIS: Fixing the bias-free final coordinate of `ln2.weight` at one and leaving its effective scaling to the unrestricted final `fc1` column will produce a 1,563-parameter model with at least 99% accuracy.

INTENDED_EDIT: Store seven learned `ln2` scale coordinates and reconstruct the eighth as a constant one during the forward pass.

EVIDENCE: The current 1,564-parameter design achieved 99.86% accuracy, while deeper `ln1` and `ln2` bias restrictions failed. This tests an independent exact affine redundancy: the current `ln2` bias basis makes its final channel identically bias-free, and the corresponding `fc1` column is unrestricted.

<<<<<<< SEARCH
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # Constant post-normalization shifts are absorbable by the
=======
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # The compact bias basis below leaves the final channel bias-free.
        # Fix its scale at one; the unrestricted final fc1 column represents
        # every effective scaling of that normalized channel.
        self.ln2.weight = nn.Parameter(
            self.ln2.weight[:-1].detach().clone()
        )

        # Constant post-normalization shifts are absorbable by the
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            self.ln2.weight,
            ln2_bias,
            self.ln2.eps,
        )
=======
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        ln2_weight = torch.cat(
            (self.ln2.weight, self.ln2.weight.new_ones(1))
        )
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            ln2_weight,
            ln2_bias,
            self.ln2.eps,
        )
>>>>>>> REPLACE