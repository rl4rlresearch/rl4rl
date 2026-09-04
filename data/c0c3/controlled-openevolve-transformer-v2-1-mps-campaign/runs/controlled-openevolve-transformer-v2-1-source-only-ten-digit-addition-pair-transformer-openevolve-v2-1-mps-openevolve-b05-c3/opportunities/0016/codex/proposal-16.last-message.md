MECHANISM: Gauge-quotient attention output bias with full-space-equivalent optimization

HYPOTHESIS: Removing the common-channel degree of freedom from each attention output-projection bias will produce 1600 parameters and retain at least 99% accuracy, because that shift passes unchanged through residual paths and is canceled by subsequent LayerNorms.

INTENDED_EDIT: Represent each eight-coordinate attention projection bias with seven learned coordinates, reconstruct a zero final coordinate during forward, and include it with the positional embedding in gauge-aware AdamW and gradient clipping.

EVIDENCE: The current four-query-bias design achieved 99.96% at 1601 parameters, while the positional quotient succeeded only after preserving full-space AdamW and clipping dynamics; this motivates applying the same qualified treatment to another exact LayerNorm-invisible direction instead of repeating the failed three-query-coordinate ablation.

<<<<<<< SEARCH
        # Construct the baseline layer first to preserve the proven constructor
        # RNG stream. Key and value biases are omitted, and only the first four
        # query coordinates are learned; all weights retain baseline shapes.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct baseline-shaped layers first to preserve the proven
        # constructor RNG stream. Key/value biases are omitted, only the first
        # four query coordinates are learned, and the output bias is represented
        # modulo its LayerNorm-invisible common-channel direction.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on an embedding quotient with one virtual gauge coordinate."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            [parameter],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )
=======
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on quotient parameters with one virtual gauge coordinate each."""

    def __init__(
        self,
        parameters: Tuple[torch.nn.Parameter, ...],
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            parameters,
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_gauge(
    model: torch.nn.Module,
    gauge_parameter: torch.nn.Parameter,
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=gauge_parameter.device, dtype=torch.float32
    )
=======
def clip_grad_norm_with_gauge(
    model: torch.nn.Module,
    gauge_parameters: Tuple[torch.nn.Parameter, ...],
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=gauge_parameters[0].device, dtype=torch.float32
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if parameter is gauge_parameter:
            # Shift invariance implies the omitted full gradient coordinate.
            total_sq.add_(grad.sum(dim=-1).square().sum())
=======
        if any(parameter is gauge for gauge in gauge_parameters):
            # Shift invariance implies the omitted full gradient coordinate.
            total_sq.add_(grad.sum(dim=-1).square().sum())
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameter = model.pos_emb.weight
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not gauge_parameter
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeAdamW(
        gauge_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    gauge_parameters = (model.pos_emb.weight,) + tuple(
        block.attn.proj.bias for block in model.blocks
    )
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if not any(parameter is gauge for gauge in gauge_parameters)
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_gauge(
                model, gauge_parameter, train_cfg.grad_clip
            )
=======
            clip_grad_norm_with_gauge(
                model, gauge_parameters, train_cfg.grad_clip
            )
>>>>>>> REPLACE