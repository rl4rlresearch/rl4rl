MECHANISM: Additional disjoint intra-head value-basis gauge fixing

HYPOTHESIS: Reproducing the qualified 1,615-parameter design and gauge-fixing a second disjoint value-channel pair in the first head will yield 1,614 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add the qualified projection-bias and seventh positional quotients, compact one value coefficient per head, and remove one additional coefficient through an independently compensated rotation of the first head’s remaining value-channel pair.

EVIDENCE: Reference Design 3 achieved 99.89% at 1,615 parameters after independent value-basis gauge fixes in both heads; this motivates extending the same exact symmetry to a disjoint channel pair instead of repeating positional or MLP quotients that failed sharply.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from attention softmax. A shared value
        # bias passes unchanged through attention and is absorbable by the
        # retained output-projection bias, so store only the query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        if self.head_dim < 2:
            raise ValueError("value-basis compaction requires head dimension at least two")
        self.d_model = d_model

        # Fix one value-basis direction per head, plus a second disjoint
        # direction in the first head when four value channels are available.
        value_basis_pairs = [(head, 0) for head in range(n_head)]
        if self.head_dim >= 4:
            value_basis_pairs.append((0, 2))
        self.value_basis_pairs = tuple(sorted(value_basis_pairs))
        self.value_fixed_indices = tuple(
            (2 * d_model + head * self.head_dim + local) * d_model
            for head, local in self.value_basis_pairs
        )

        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from attention softmax. A shared value
        # bias passes unchanged through attention and is absorbable by the
        # retained output-projection bias, so store only the query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)

        # A uniform projection-bias shift survives the residual connection but
        # is erased by both subsequent LayerNorms. Store only its zero-sum part.
        self.proj.bias = nn.Parameter(self.proj.bias.new_zeros(d_model - 1))
        proj_bias_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            proj_bias_basis[: col + 1, col] = 1.0 / scale
            proj_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("proj_bias_basis", proj_bias_basis, persistent=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compact_value_basis(self) -> None:
        # Attention mixes value coordinates identically within each head.
        # Rotate each selected pair, counter-rotate the corresponding output
        # columns, and omit the coefficient made exactly zero.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for head, local in self.value_basis_pairs:
                first_value = 2 * self.d_model + head * self.head_dim + local
                second_value = first_value + 1
                first_column = head * self.head_dim + local
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
        bsz, seqlen, d_model = x.shape
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
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_bias = self.proj_bias_basis @ self.proj.bias
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
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
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
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