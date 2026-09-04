MECHANISM: Intra-head value-basis gauge fixing

HYPOTHESIS: Fixing one value-projection coefficient through an exactly compensated orthogonal rotation will reduce the verified model from 1,617 to 1,616 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate two value channels and their corresponding output-projection columns after initialization, omit the resulting zero coefficient, and reconstruct it during every forward pass.

EVIDENCE: The current 1,617-parameter design achieved 100%, whereas three attempts to reach 1,616 by compacting another positional row scored 88.94%, 72.89%, and 74.11%; this tests an independent exact symmetry without adding another sensitive positional constraint.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
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
        # Attention applies the same mixing matrix to every value coordinate
        # within a head. Rotate two such coordinates, compensate with the
        # inverse rotation in the output projection, and gauge-fix one value
        # coefficient to zero without changing the initialized function.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()
            first_value = 2 * self.d_model
            second_value = first_value + 1

            a = qkv_weight[first_value, 0]
            b = qkv_weight[second_value, 0]
            norm = torch.hypot(a, b)
            cosine = b / norm
            sine = a / norm

            row0 = qkv_weight[first_value].clone()
            row1 = qkv_weight[second_value].clone()
            qkv_weight[first_value] = cosine * row0 - sine * row1
            qkv_weight[second_value] = sine * row0 + cosine * row1

            col0 = proj_weight[:, 0].clone()
            col1 = proj_weight[:, 1].clone()
            proj_weight[:, 0] = cosine * col0 - sine * col1
            proj_weight[:, 1] = sine * col0 + cosine * col1
            self.proj.weight.copy_(proj_weight)

            flat_weight = qkv_weight.reshape(-1)
            fixed_index = first_value * self.d_model
            compact_weight = torch.cat(
                (flat_weight[:fixed_index], flat_weight[fixed_index + 1 :])
            )
        self.qkv.weight = nn.Parameter(compact_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
        fixed_index = 2 * self.d_model * self.d_model
        qkv_weight = torch.cat(
            (
                self.qkv.weight[:fixed_index],
                self.qkv.weight.new_zeros(1),
                self.qkv.weight[fixed_index:],
            )
        ).view(3 * self.d_model, self.d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()

        # Preserve baseline initialization and RNG consumption, then remove
>>>>>>> REPLACE