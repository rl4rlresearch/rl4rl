MECHANISM: Batched terminal-MLP output gauge quotient with verified three-row key compaction

HYPOTHESIS: Compacting all 12 columns of the terminal MLP output weight into the seven-dimensional zero-sum residual subspace, on top of the verified three-row key quotient, will produce a 1,598-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified three-row key quotient, then store the entire MLP output weight in compact output coordinates, reconstruct it during inference, and preserve virtual eight-coordinate AdamW dynamics for every compact parameter.

EVIDENCE: The three-row key design achieved 99.96% accuracy at 1,610 parameters, and the existing successful `fc2.bias` quotient establishes that uniform MLP output components are removed by downstream LayerNorm; applying the same symmetry column-wise removes 12 additional functionally null parameters.

<<<<<<< SEARCH
        self.qkv.bias = None
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv.bias = None

        # Each key row has one input common-mode coordinate that produces
        # only a position-independent key offset and cancels in softmax.
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
        qkv = F.linear(x, self.qkv.weight, None)
        q, k, v = qkv.chunk(3, dim=-1)
=======
    def compact_key_weights(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = k_weight[-3:] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-3].clone())
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
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
=======
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def compact_output_weight(self) -> None:
        with torch.no_grad():
            compact_weight = (
                self.bias_basis.transpose(0, 1) @ self.fc2.weight.detach()
            )
        self.fc2.weight = nn.Parameter(compact_weight.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        weight = self.bias_basis @ self.fc2.weight
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, weight, bias))
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
            blk.attn.compact_key_weights()
            blk.mlp.compact_output_weight()

        # Preserve baseline initialization and RNG consumption, then remove
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
=======
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (blk.mlp.fc2.weight, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (compact, blk.attn.key_basis)
        for blk in model.blocks
        for compact in blk.attn.k_compact
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in mlp_gauge_params
    ]
=======
    mlp_gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(
                (basis.size(0),) + tuple(param.shape[1:])
            ),
            "exp_avg_sq": basis.new_zeros(
                (basis.size(0),) + tuple(param.shape[1:])
            ),
        }
        for param, basis in mlp_gauge_params
    ]
>>>>>>> REPLACE