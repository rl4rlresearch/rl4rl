MECHANISM: Null-space reduction of the post-normalization MLP input map

HYPOTHESIS: The qualified 1555-parameter design plus a mean-zero input parameterization for `fc1` will achieve at least 99% accuracy with 1543 parameters, because affine-free `ln2` always supplies zero-mean features, making one common-mode weight per `fc1` output exactly unobservable.

INTENDED_EDIT: Reproduce the qualified three-gauge, folded-attention-bias, bias-free `ln1`, affine-free `ln2` design, then represent each `fc1` weight row in the seven-dimensional mean-zero basis while preserving full-size initialization draws.

EVIDENCE: The affine-free-`ln2` design achieved 99.89% at 1555 parameters. Unlike the failed removal of active `ln1` scales at 1547 parameters, this removes only the 12 `fc1` directions annihilated by affine-free LayerNorm outputs.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and two token/position shift gauges fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords[-2:]
        gauged_weight = gauged_weight - basis[:, -2:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-2])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )
=======
class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with global and three token/position shift gauges fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - basis[:, -3:] @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.last_weight = nn.Parameter(last_coords[:-3])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-2:]
        gauged_weight = gauged_weight - self.basis[:, -2:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-2])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 2))
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords[-3:]
        gauged_weight = gauged_weight - self.basis[:, -3:] @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.last_weight.copy_(last_coords[:-3])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ F.pad(self.last_weight, (0, 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
=======
class MeanZeroInputLinear(nn.Linear):
    """Linear map modulo the common input mode removed by LayerNorm."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(in_features)
        self.weight = nn.Parameter(self.weight.detach() @ basis)
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight @ self.basis.transpose(0, 1), self.bias)


class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (self.qv_bias[0], self.qkv.weight.new_zeros(d_model), self.qv_bias[1])
        )
=======
        qkv_bias = torch.cat(
            (
                self.q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            self.pos_emb.weight[:, -2:].add_(
                self.token_emb.initial_position_shift
            )
=======
        with torch.no_grad():
            self.pos_emb.weight[:, -3:].add_(
                self.token_emb.initial_position_shift
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanZeroOutputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
                module.bias.zero_()
=======
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.basis)
                module.bias.zero_()
        elif isinstance(module, MeanZeroOutputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
                module.bias.zero_()
>>>>>>> REPLACE