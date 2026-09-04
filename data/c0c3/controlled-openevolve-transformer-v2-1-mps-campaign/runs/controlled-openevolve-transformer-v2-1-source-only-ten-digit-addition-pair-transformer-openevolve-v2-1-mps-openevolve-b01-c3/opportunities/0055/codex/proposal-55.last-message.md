MECHANISM: Cross-head sharing of the four sparsest relative-distance biases

HYPOTHESIS: A 1,282-parameter model will retain at least 99% accuracy because the verified 1,283-parameter model achieved 100%, while additionally sharing the fourth-farthest bias preserves an adaptive value and affects only four query-key pairs at full context.

INTENDED_EDIT: Anchor distance-zero biases at zero and share the four maximum-distance attention biases across heads, while retaining independent head-specific biases at every shorter causal distance.

EVIDENCE: The 1,283-parameter design achieved 100% accuracy after successfully extending cross-head sharing from the two sparsest distance bins to three; sharing the next-sparsest bin is the smallest directly supported reduction.

<<<<<<< SEARCH
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 5))
        self.far_rel_bias = nn.Parameter(torch.zeros(4))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        positions = torch.arange(seqlen, device=x.device)
        relative_distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        att = att + self.rel_bias[:, relative_distance].unsqueeze(0)
=======
        positions = torch.arange(seqlen, device=x.device)
        relative_distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        learned_bias = torch.cat(
            (
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 4),
            ),
            dim=1,
        )
        learned_bias = F.pad(learned_bias, (1, 0))
        att = att + learned_bias[:, relative_distance].unsqueeze(0)
>>>>>>> REPLACE