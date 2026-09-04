MECHANISM: Residually gated nonlinear channel

HYPOTHESIS: An 8-channel multiplicatively gated MLP will retain at least 99% accuracy with 1,624 parameters because it preserves the width-12 MLP’s 192 learned weight coefficients while replacing four additive GELU channels with input-dependent feature interactions suited to carry decisions.

INTENDED_EDIT: Start from the verified key-bias-free and `ln2`-bias-free attention block, then replace the width-12 additive MLP with an 8-channel GELU pathway modulated by a learned bias-free gate.

EVIDENCE: The 1,628-parameter design reached 99.95%, while simply narrowing its additive MLP to eight channels is contraindicated by the 62.9% result. This patch challenges the load-bearing assumption that 12 additive channels are necessary by preserving the MLP weight budget and changing the learned computation to multiplicative gating.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x)) * (1.0 + self.gate(x))
        return self.drop(self.fc2(hidden))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE