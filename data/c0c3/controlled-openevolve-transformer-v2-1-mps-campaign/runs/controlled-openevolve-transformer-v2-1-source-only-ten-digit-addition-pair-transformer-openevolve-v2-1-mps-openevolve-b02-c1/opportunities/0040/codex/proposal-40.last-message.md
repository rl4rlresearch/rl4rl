MECHANISM: Complementary signed relative-position potential

HYPOTHESIS: Replacing the two independent per-head distance tables with one learned positional potential, read positively by one head and with a learned initially negative scale by the other, will reduce the model from 1,173 to 1,152 parameters while retaining at least 99% accuracy because its maxima and minima can encode complementary attention landmarks.

INTENDED_EDIT: Challenge the assumption that each attention head requires an unconstrained positional table: share one gauge-fixed relative-bias vector, fix the first head’s scale to one, and learn the second head’s polarity and magnitude.

EVIDENCE: The current two-head model achieves 99.98% with only nine MLP units, indicating optimization margin. The successful Q/K and value/output basis quotients also show that attention-head roles survive substantial structural reparameterization, motivating a direct test of complementary rather than independent positional specialization.

<<<<<<< SEARCH
        # One bias per head is a softmax-invisible common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        # Heads read complementary extrema of one learned positional
        # potential. Fixing the first scale removes the product's scale gauge.
        self.rel_bias = nn.Parameter(torch.zeros(1, max_seq_len - 1))
        self.rel_scale = nn.Parameter(-torch.ones(n_head - 1, 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        rel_bias = F.pad(self.rel_bias, (0, 1))
        att = att + rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
=======
        rel_scale = torch.cat(
            [self.rel_bias.new_ones(1, 1), self.rel_scale], dim=0
        )
        rel_bias = rel_scale * F.pad(self.rel_bias, (0, 1))
        att = att + rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
>>>>>>> REPLACE