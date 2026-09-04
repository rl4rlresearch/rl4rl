MECHANISM: Absorb the second LayerNorm bias into the MLP input bias

HYPOTHESIS: The 1628-parameter qualified design can remove all eight `ln2` bias parameters and retain at least 99% accuracy, because `fc1.bias` can represent the LayerNorm bias’s entire downstream effect without reducing the function class, producing 1620 parameters.

INTENDED_EDIT: Apply the qualified query-only QKV bias design, retain `d_ff=12`, and remove the second LayerNorm bias while preserving the proven initialization RNG stream.

EVIDENCE: The query-only QKV design achieved 99.92% accuracy at 1628 parameters; unlike the failed `d_ff` reductions, this removes an analytically redundant affine parameter while preserving nonlinear capacity.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Construct with the baseline shape first to preserve the qualified
        # initialization RNG stream. Only query bias is functionally needed.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        bias = torch.cat((self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model)))
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # Any ln2 bias can be absorbed into fc1.bias after the linear map.
        self.ln2.bias = None
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12"))
=======
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12_qbias_noln2bias")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12_qbias_noln2bias"))
>>>>>>> REPLACE