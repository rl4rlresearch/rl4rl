MECHANISM: Two-row key-projection gauge quotient

HYPOTHESIS: Extending the verified single-row key quotient to the adjacent penultimate key row will produce a 1,611-parameter model with at least 99% accuracy, because each removed coordinate independently contributes only a position-constant key offset that cancels inside attention softmax.

INTENDED_EDIT: Adopt the verified bias-free attention and final-four positional quotient, then store the final two key-projection rows in independent seven-dimensional zero-sum bases with reconstructed full-coordinate AdamW updates.

EVIDENCE: The 1,612-parameter single-key-row quotient achieved 100% accuracy, while the alternative 1,612-parameter attention-output-bias quotient missed at 98.87%; incrementally quotienting a second key row therefore tests the strongest demonstrated reduction mechanism.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from every attention-softmax row. Retain
        # the original parameter slot while storing only query/value biases.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve constructor initialization while omitting all QKV biases.
        self.qkv.bias = None

        # For each compacted key row, one input common-mode coordinate only
        # produces a position-independent key offset and cancels in softmax.
        key_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            key_basis[: col + 1, col] = 1.0 / scale
            key_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("key_basis", key_basis, persistent=False)

        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compact_key_weights(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = k_weight[-2:] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-2].clone())
        self.k_compact = nn.ParameterList(
            [nn.Parameter(row.clone()) for row in compact_key]
        )
        self.v_weight = nn.Parameter(v_weight.clone())
        del self.qkv

    def forward(self, x: torch.Tensor, key_x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q = F.linear(x, self.q_weight)
        key_main = F.linear(x, self.k_weight)
        key_tail = F.linear(
            key_x @ self.key_basis,
            torch.stack(tuple(self.k_compact), dim=0),
        )
        k = torch.cat((key_main, key_tail), dim=-1)
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
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final three positions, which have the shortest causal influence.
        self.compact_pos_count = 5
=======
        self.apply(self._init_weights)
        for blk in self.blocks:
            blk.attn.compact_key_weights()

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
        self.compact_pos_count = 6
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-3:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-3].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
=======
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (compact, blk.attn.key_basis)
        for blk in model.blocks
        for compact in blk.attn.k_compact
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-3:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-3].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
>>>>>>> REPLACE