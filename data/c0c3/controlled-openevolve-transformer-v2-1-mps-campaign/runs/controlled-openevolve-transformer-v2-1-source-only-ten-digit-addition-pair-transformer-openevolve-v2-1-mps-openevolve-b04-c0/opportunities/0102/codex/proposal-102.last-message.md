MECHANISM: Mean-zero folded attention output bias

HYPOTHESIS: Folding the tied value/projection bias into a seven-parameter mean-zero post-projection bias will produce a 1576-parameter model with at least 99% accuracy, because with zero dropout each attention row sums to one, while the final LayerNorm removes the remaining uniform-shift degree of freedom.

INTENDED_EDIT: Replace the eight-parameter bias shared by the value and output projections with seven learned coordinates that reconstruct a zero-sum attention output bias.

EVIDENCE: The current 1577-parameter design reached 99.13%, whereas the multiplicative value/output weight gauge reached only 55.37%; this additive reparameterization preserves the zero initialization and avoids reciprocal scaling.

<<<<<<< SEARCH
        self.qkv = AttentionWeightAnchoredLinear(d_model, 3 * d_model)
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = self.v_bias
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = AttentionWeightAnchoredLinear(d_model, 3 * d_model)
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.attn_bias_rest = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = None
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        q = q + q_bias
        v = v + self.v_bias
=======
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        q = q + q_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        attn_bias = torch.cat(
            (self.attn_bias_rest, -self.attn_bias_rest.sum().unsqueeze(0))
        )
        y = F.linear(y, self.proj.weight, attn_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE