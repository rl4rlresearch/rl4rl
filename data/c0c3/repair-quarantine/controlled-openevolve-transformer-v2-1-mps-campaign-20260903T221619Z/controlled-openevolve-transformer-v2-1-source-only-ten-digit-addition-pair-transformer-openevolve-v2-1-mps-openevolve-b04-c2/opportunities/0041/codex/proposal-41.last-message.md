MECHANISM: Per-head value-basis gauge fixing

HYPOTHESIS: Reproducing the qualified 1,616-parameter design while applying the same compensated value-channel rotation independently to the second attention head will yield 1,615 parameters and at least 99% accuracy.

INTENDED_EDIT: Add the qualified `ln1.bias` and seventh positional quotients, then rotate and omit one value-projection coefficient per attention head while exactly counter-rotating the corresponding output-projection columns.

EVIDENCE: Reference Design 2 achieved 99.99% accuracy at 1,616 parameters with one value-basis gauge fix; that reduction succeeded where repeated additional positional quotients failed, motivating one independent application of the same exact symmetry in the other head.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        if self.head_dim < 2:
            raise ValueError("value-basis compaction requires head dimension at least two")
        self.d_model = d_model
        self.value_fixed_indices = tuple(
            (2 * d_model + head * self.head_dim) * d_model
            for head in range(n_head)
        )
        self.qkv = nn.Linear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compact_value_basis(self) -> None:
        # Attention mixes value coordinates identically within each head.
        # Rotate two coordinates per head, counter-rotate the corresponding
        # output columns, and gauge-fix one coefficient in each rotation.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for head in range(self.n_head):
                first_value = 2 * self.d_model + head * self.head_dim
                second_value = first_value + 1
                first_column = head * self.head_dim
                second_column = first_column + 1

                a = qkv_weight[first_value, 0]
                b = qkv_weight[second_value, 0]
                norm = torch.hypot(a, b)
                cosine = b / norm
                sine = a / norm

                row0 = qkv_weight[first_value].clone()
                row1 = qkv_weight[second_value].clone()
                qkv_weight[first_value] = cosine * row0 - sine * row1
                qkv_weight[second_value] = sine * row0 + cosine * row1

                col0 = proj_weight[:, first_column].clone()
                col1 = proj_weight[:, second_column].clone()
                proj_weight[:, first_column] = cosine * col0 - sine * col1
                proj_weight[:, second_column] = sine * col0 + cosine * col1

            self.proj.weight.copy_(proj_weight)
            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
                pieces.append(flat_weight[start:fixed_index])
                start = fixed_index + 1
            pieces.append(flat_weight[start:])
            compact_weight = torch.cat(pieces)
        self.qkv.weight = nn.Parameter(compact_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
        weight_pieces = []
        compact_start = 0
        for removed, fixed_index in enumerate(self.value_fixed_indices):
            compact_index = fixed_index - removed
            weight_pieces.append(self.qkv.weight[compact_start:compact_index])
            weight_pieces.append(self.qkv.weight.new_zeros(1))
            compact_start = compact_index
        weight_pieces.append(self.qkv.weight[compact_start:])
        qkv_weight = torch.cat(weight_pieces).view(
            3 * self.d_model, self.d_model
        )
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # A uniform pre-attention shift is absorbable by the query and
        # output-projection biases; key shifts cancel from attention softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ln1_bias = self.ln1_bias_basis @ self.ln1.bias
        normalized = F.layer_norm(
            x,
            self.ln1.normalized_shape,
            self.ln1.weight,
            ln1_bias,
            self.ln1.eps,
        )
        x = x + self.attn(normalized)

        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
        self.compact_pos_count = 6
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final five positions, which have the shortest causal influence.
        self.compact_pos_count = 7
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-5:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-5].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
        )
=======
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-5:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-5].reshape(-1),
                    )
                )
>>>>>>> REPLACE