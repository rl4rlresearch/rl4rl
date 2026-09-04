MECHANISM: Parameter-free second pre-norm with shared key/value attention

HYPOTHESIS: The verified shared-key/value, bias-free-value design will retain at least 99% accuracy after removing all eight `ln2` scale parameters, yielding 1,542 parameters, because `ln2` scales initialize to one and their per-coordinate scaling can be learned equivalently by the immediately following `fc1` columns.

INTENDED_EDIT: Replace full key/value heads with one shared four-dimensional learned key/value stream, omit the redundant value bias, and make the second pre-norm non-affine.

EVIDENCE: The shared-key/value design without value bias achieved 99.85% accuracy at 1,550 parameters; this preserves that qualified computation while removing only the scale directly preceding a learned affine layer.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # Preserve distinct learned queries while sharing one learned
        # key/value representation across both attention maps.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
        # The singleton head dimension broadcasts the shared learned streams
        # across the independently parameterized query heads.
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE