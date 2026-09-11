MECHANISM: Bias-only feed-forward limit

HYPOTHESIS: Replacing the final one-neuron MLP with its learned residual bias will reduce parameters from 1,053 to 1,038 while retaining at least 99% accuracy, because every preceding one-neuron ablation succeeded and the width-1 model achieved 100%.

INTENDED_EDIT: Remove the last nonlinear feed-forward unit while retaining its seven-parameter residual bias, preserve the initialization RNG stream, and remove the deleted projection from the quotient optimizer.

EVIDENCE: The width-1 design achieved 100% accuracy after eleven consecutive successful width reductions; a bias-only branch continues the same 15-parameter ablation while isolating whether any nonlinear MLP unit remains necessary.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2._residual_gauge = True

        # Preserve the RNG stream of the original full-width Linear constructor.
        bound = 1.0 / math.sqrt(d_ff)
        torch.empty(d_ff).uniform_(-bound, bound)

        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
=======
class MLP(nn.Module):
    """Bias-only limit of the feed-forward residual branch."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        # Preserve the constructor RNG stream of the removed Linear modules.
        discarded_fc1 = torch.empty(d_ff, d_model - 1)
        nn.init.kaiming_uniform_(discarded_fc1, a=math.sqrt(5))
        fc1_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(-fc1_bound, fc1_bound)

        discarded_fc2 = torch.empty(d_model - 1, d_ff)
        nn.init.kaiming_uniform_(discarded_fc2, a=math.sqrt(5))
        fc2_bound = 1.0 / math.sqrt(d_ff)
        torch.empty(d_ff).uniform_(-fc2_bound, fc2_bound)

        self._removed_mlp_shapes = (
            (d_ff, d_model - 1),
            (d_model, d_ff),
        )
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.pad(self.fc2_bias, (0, 1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        removed_pos_shape = getattr(module, "_removed_pos_shape", None)
        if removed_pos_shape is not None:
            torch.empty(removed_pos_shape).normal_()

        if isinstance(module, GaugeFixedQKV):
=======
        removed_pos_shape = getattr(module, "_removed_pos_shape", None)
        if removed_pos_shape is not None:
            torch.empty(removed_pos_shape).normal_()

        removed_mlp_shapes = getattr(module, "_removed_mlp_shapes", None)
        if removed_mlp_shapes is not None:
            for shape in removed_mlp_shapes:
                torch.empty(shape).normal_(mean=0.0, std=0.02)

        if isinstance(module, GaugeFixedQKV):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
=======
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
>>>>>>> REPLACE