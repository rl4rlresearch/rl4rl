MECHANISM: Fused four-row key gauge quotient with single-pass normalization

HYPOTHESIS: A 1,609-parameter four-row key quotient will complete within the time limit and retain at least 99% accuracy when its compact rows are batched, the remaining Q/K/V projections are fused, redundant LayerNorm computation is eliminated, and validation frequency is reduced.

INTENDED_EDIT: Extend key compaction from three to four rows, store all compact rows in one matrix with batched virtual AdamW updates, fuse the other attention projections, compute normalization once per block, and evaluate every 1,000 steps.

EVIDENCE: Three compacted key rows achieved 99.96% at 1,610 parameters; both four-row attempts timed out without an accuracy failure, so the strongest supported next test is the same one-parameter reduction with additional execution savings.

<<<<<<< SEARCH
        # For each compacted key row, one input common-mode coordinate only
        # produces a position-independent key offset and cancels in softmax.
        key_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            key_basis[: col + 1, col] = 1.0 / scale
            key_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("key_basis", key_basis, persistent=False)

        self.proj = nn.Linear(d_model, d_model)
=======
        # For each compacted key row, one input common-mode coordinate only
        # produces a position-independent key offset and cancels in softmax.
        self.compact_key_rows = 4
        key_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            key_basis[: col + 1, col] = 1.0 / scale
            key_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("key_basis", key_basis, persistent=False)

        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def compact_key_weights(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = (
                self.key_basis.transpose(0, 1)
                @ k_weight[-self.compact_key_rows :].transpose(0, 1)
            )
            main_weight = torch.cat(
                (
                    q_weight,
                    k_weight[: -self.compact_key_rows],
                    v_weight,
                ),
                dim=0,
            )
        self.main_weight = nn.Parameter(main_weight.clone())
        self.k_compact = nn.Parameter(compact_key.clone())
        del self.qkv

    def forward(self, x: torch.Tensor, key_x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        main = F.linear(x, self.main_weight)
        q, key_main, v = torch.split(
            main,
            (d_model, d_model - self.compact_key_rows, d_model),
            dim=-1,
        )
        key_tail_weight = (
            self.key_basis @ self.k_compact
        ).transpose(0, 1)
        key_tail = F.linear(key_x, key_tail_weight)
        k = torch.cat((key_main, key_tail), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.ln1(x)
        key_input = F.layer_norm(
            x, (x.size(-1),), eps=self.ln1.eps
        )
        x = x + self.attn(attn_input, key_input)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key_input = F.layer_norm(
            x, (x.size(-1),), eps=self.ln1.eps
        )
        attn_input = key_input * self.ln1.weight + self.ln1.bias
        x = x + self.attn(attn_input, key_input)
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (compact, blk.attn.key_basis)
        for blk in model.blocks
        for compact in blk.attn.k_compact
    ]
=======
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (blk.attn.k_compact, blk.attn.key_basis) for blk in model.blocks
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

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE