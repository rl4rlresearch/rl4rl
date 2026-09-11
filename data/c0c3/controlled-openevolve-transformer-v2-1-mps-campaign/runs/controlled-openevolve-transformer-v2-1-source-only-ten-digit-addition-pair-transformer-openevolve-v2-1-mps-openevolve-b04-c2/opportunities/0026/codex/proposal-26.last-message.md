MECHANISM: Single-row key-projection gauge quotient

HYPOTHESIS: Removing the verified redundant QKV biases and quotienting one key-projection row into a seven-dimensional zero-sum basis will produce a 1,612-parameter model with at least 99% accuracy.

INTENDED_EDIT: Make attention bias-free, store one key row without its LayerNorm-induced constant-offset coordinate, reconstruct its computation from the un-affined normalized residual stream, and preserve virtual eight-coordinate AdamW dynamics.

EVIDENCE: The bias-free 1,613-parameter design achieved 99.84%; the failed 1,612-parameter output-bias quotient reached 98.87%, motivating a distinct exact attention symmetry where a key component constant across positions cancels from every softmax row.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from every attention-softmax row. A value
        # bias also passes unchanged through attention because each row sums
        # to one, so the output-projection bias can absorb it exactly. Retain
        # only the learned query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve constructor initialization while omitting all QKV biases.
        self.qkv.bias = None

        # One coordinate of a key row only produces a position-independent
        # key offset after LayerNorm and therefore cancels from the softmax.
        key_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            key_basis[: col + 1, col] = 1.0 / scale
            key_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("key_basis", key_basis, persistent=False)

        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        zero_bias = torch.zeros_like(q_bias)
        qkv_bias = torch.cat((q_bias, zero_bias, zero_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
    def compact_key_weight(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = k_weight[-1] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-1].clone())
        self.k_compact = nn.Parameter(compact_key.clone())
        self.v_weight = nn.Parameter(v_weight.clone())
        del self.qkv

    def forward(self, x: torch.Tensor, key_x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q = F.linear(x, self.q_weight)
        key_main = F.linear(x, self.k_weight)
        key_last = F.linear(
            key_x @ self.key_basis, self.k_compact.unsqueeze(0)
        )
        k = torch.cat((key_main, key_last), dim=-1)
        v = F.linear(x, self.v_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.ln1(x)
        key_input = F.layer_norm(
            x, (x.size(-1),), eps=self.ln1.eps
        )
        x = x + self.attn(attn_input, key_input)
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
=======
        self.apply(self._init_weights)
        for blk in self.blocks:
            blk.attn.compact_key_weight()

        # Preserve baseline initialization and RNG consumption, then remove
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
=======
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (blk.attn.k_compact, blk.attn.key_basis) for blk in model.blocks
    ]
>>>>>>> REPLACE