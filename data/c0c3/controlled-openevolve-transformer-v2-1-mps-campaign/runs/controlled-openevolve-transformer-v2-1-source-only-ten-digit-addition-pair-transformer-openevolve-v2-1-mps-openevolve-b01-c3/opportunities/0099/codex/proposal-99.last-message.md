MECHANISM: Pair-tied MLP output bias with unfactorized headwise attention maps

HYPOTHESIS: The 1,136-parameter transformer will maintain at least 99% accuracy because the verified 1,137-parameter design reached 99.92%, and this ties only one pair of initially identical MLP output-bias coordinates while preserving every learned weight matrix and attention mechanism.

INTENDED_EDIT: Adopt the verified fixed-direction scalar addressing, direct per-head 8→7 attention output maps, and bias-free MLP expansion, then reduce one additional parameter by pair-tying the mean-zero MLP output bias.

EVIDENCE: Reference Design 1 achieved 99.92% with 1,137 parameters. The successful 1,145-parameter result also showed that removing an initially zero MLP hidden bias preserved 100% accuracy, motivating another isolated MLP-bias reduction instead of altering the repeatedly sensitive positional biases or final normalization.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query_dim = 1
        self.address = nn.Linear(d_model, self.query_dim, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
=======
        self.n_head = n_head
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
        address = self.address(x)
        v = self.value(x)
=======
        address = self.address_scale * x[..., :1]
        v = self.value(x)
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
        self.fc2 = MeanZeroLinear(d_ff, d_model)
=======
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = PairTiedBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE