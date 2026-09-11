MECHANISM: Unfactorized headwise value-output maps

HYPOTHESIS: The 1,137-parameter transformer will achieve at least 99% accuracy because each head’s factorized 8→4→7 value/output path can be replaced by a more expressive direct 8→7 learned map while preserving attention, positional routing, normalization, and residual biases.

INTENDED_EDIT: Adopt the verified bias-free MLP and fuse the attention value and output weights into direct per-head mean-zero output maps, removing eight factorization parameters.

EVIDENCE: The verified 1,145-parameter bias-free design achieved 100% accuracy; prior failures targeted positional biases or normalization, whereas this patch preserves those mechanisms and removes only a low-rank factorization whose direct replacement has fewer parameters and no smaller effective linear function class.

<<<<<<< SEARCH
        self.query_dim = 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
=======
        self.query_dim = 1
        self.output_dim = d_model - 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = nn.Linear(
            d_model, n_head * self.output_dim, bias=False
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
        self.register_buffer(
            "output_basis", mean_zero_basis(d_model), persistent=False
        )
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        v = v.view(bsz, seqlen, self.n_head, self.output_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.sum(dim=1)
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(2)))
        y = (y + bias) @ self.output_basis.transpose(0, 1)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = PairwiseTiedBiasLinear(d_model, d_ff)
=======
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
>>>>>>> REPLACE