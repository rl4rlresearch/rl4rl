MECHANISM: Packed full-MLP output-shift gauge

HYPOTHESIS: Packing all twelve zero-anchored `fc2` columns into one grouped gauge parameter will reduce the model from 1,573 to 1,572 parameters, retain at least 99% accuracy after 21,000 steps, and finish within the verification limit.

INTENDED_EDIT: Store `fc2.weight` as one 7-by-12 relative-coordinate matrix, reconstruct its implicit zero output row in one operation, update its twelve gauges together, and eliminate avoidable scalar tensor accesses in sampling.

EVIDENCE: The verified 1,573-parameter model reached 99.98% while independently anchoring eleven of twelve `fc2` columns; prior final-column attempts timed out, motivating the same exact symmetry in a denser representation with fewer forward and optimizer dispatches.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2._defer_two_column_gauge = True
        self.fc2.register_parameter("unused", None)
        self.register_parameter("fc2_col_5", None)
        self.register_parameter("fc2_col_4", None)
        self.register_parameter("fc2_col_3", None)
        self.register_parameter("fc2_col_2", None)
        self.register_parameter("fc2_col", None)
        self.register_parameter("fc2_col_0", None)
        self.register_parameter("fc2_col_1", None)
        self.register_parameter("fc2_col_low", None)
        self.register_parameter("fc2_col_mid", None)
        self.register_parameter("fc2_col_4_abs", None)
        self.register_parameter("fc2_col_5_abs", None)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = torch.cat(
            (self.fc1.weight, self.fc1.weight.new_zeros(1))
        ).view(self.fc1.out_features, self.fc1.in_features)
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_col_0 = torch.cat(
            (self.fc2_col_0, self.fc2_col_0.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_1 = torch.cat(
            (self.fc2_col_1, self.fc2_col_1.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_low = torch.cat(
            (self.fc2_col_low, self.fc2_col_low.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_mid = torch.cat(
            (self.fc2_col_mid, self.fc2_col_mid.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_4_abs = torch.cat(
            (self.fc2_col_4_abs, self.fc2_col_4_abs.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_5_abs = torch.cat(
            (self.fc2_col_5_abs, self.fc2_col_5_abs.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_5 = torch.cat(
            (self.fc2_col_5, self.fc2_col_5.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_4 = torch.cat(
            (self.fc2_col_4, self.fc2_col_4.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_3 = torch.cat(
            (self.fc2_col_3, self.fc2_col_3.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_2 = torch.cat(
            (self.fc2_col_2, self.fc2_col_2.new_zeros(1))
        ).unsqueeze(1)
        fc2_col = torch.cat(
            (self.fc2_col, self.fc2_col.new_zeros(1))
        ).unsqueeze(1)
        fc2_weight = torch.cat(
            (
                fc2_col_0,
                fc2_col_1,
                fc2_col_low,
                fc2_col_mid,
                fc2_col_4_abs,
                fc2_col_5_abs,
                self.fc2.weight,
                fc2_col_5,
                fc2_col_4,
                fc2_col_3,
                fc2_col_2,
                fc2_col,
            ),
            dim=1,
        )
        fc2_bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.weight = nn.Parameter(torch.empty(d_model - 1, d_ff))
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = torch.cat(
            (self.fc1.weight, self.fc1.weight.new_zeros(1))
        ).view(self.fc1.out_features, self.fc1.in_features)
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_weight = torch.cat(
            (
                self.fc2.weight,
                self.fc2.weight.new_zeros(1, self.fc2.in_features),
            ),
            dim=0,
        )
        fc2_bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_fc2_col = block.mlp.fc2.weight.new_empty(cfg.d_model)
                nn.init.normal_(full_fc2_col, mean=0.0, std=0.02)
                full_fc2_col_0 = block.mlp.fc2.weight[:, 0].detach().clone()
                full_fc2_col_1 = block.mlp.fc2.weight[:, 1].detach().clone()
                full_fc2_col_low = block.mlp.fc2.weight[:, 2].detach().clone()
                full_fc2_col_mid = block.mlp.fc2.weight[:, 3].detach().clone()
                full_fc2_col_4_abs = block.mlp.fc2.weight[:, 4].detach().clone()
                full_fc2_col_5_abs = block.mlp.fc2.weight[:, 5].detach().clone()
                full_fc2_col_5 = block.mlp.fc2.weight[:, -5].detach().clone()
                full_fc2_col_4 = block.mlp.fc2.weight[:, -4].detach().clone()
                full_fc2_col_3 = block.mlp.fc2.weight[:, -3].detach().clone()
                full_fc2_col_2 = block.mlp.fc2.weight[:, -2].detach().clone()
                base_fc2_weight = block.mlp.fc2.weight[:, 6:-5].detach().clone()
                block.mlp.fc2.weight = nn.Parameter(base_fc2_weight)

                full_fc2_col_0.sub_(full_fc2_col_0[-1].clone())
                block.mlp.fc2_col_0 = nn.Parameter(
                    full_fc2_col_0[:-1].clone()
                )
                full_fc2_col_1.sub_(full_fc2_col_1[-1].clone())
                block.mlp.fc2_col_1 = nn.Parameter(
                    full_fc2_col_1[:-1].clone()
                )
                full_fc2_col_low.sub_(full_fc2_col_low[-1].clone())
                block.mlp.fc2_col_low = nn.Parameter(
                    full_fc2_col_low[:-1].clone()
                )
                full_fc2_col_mid.sub_(full_fc2_col_mid[-1].clone())
                block.mlp.fc2_col_mid = nn.Parameter(
                    full_fc2_col_mid[:-1].clone()
                )
                full_fc2_col_4_abs.sub_(full_fc2_col_4_abs[-1].clone())
                block.mlp.fc2_col_4_abs = nn.Parameter(
                    full_fc2_col_4_abs[:-1].clone()
                )
                full_fc2_col_5_abs.sub_(full_fc2_col_5_abs[-1].clone())
                block.mlp.fc2_col_5_abs = nn.Parameter(
                    full_fc2_col_5_abs[:-1].clone()
                )
                full_fc2_col_5.sub_(full_fc2_col_5[-1].clone())
                block.mlp.fc2_col_5 = nn.Parameter(
                    full_fc2_col_5[:-1].clone()
                )
                full_fc2_col_4.sub_(full_fc2_col_4[-1].clone())
                block.mlp.fc2_col_4 = nn.Parameter(
                    full_fc2_col_4[:-1].clone()
                )
                full_fc2_col_3.sub_(full_fc2_col_3[-1].clone())
                block.mlp.fc2_col_3 = nn.Parameter(
                    full_fc2_col_3[:-1].clone()
                )
                full_fc2_col_2.sub_(full_fc2_col_2[-1].clone())
                block.mlp.fc2_col_2 = nn.Parameter(
                    full_fc2_col_2[:-1].clone()
                )
                full_fc2_col.sub_(full_fc2_col[-1].clone())
                block.mlp.fc2_col = nn.Parameter(full_fc2_col[:-1].clone())
=======
                full_fc2_weight = block.mlp.fc2.weight.new_empty(
                    cfg.d_model, cfg.d_ff
                )
                nn.init.normal_(full_fc2_weight, mean=0.0, std=0.02)
                full_fc2_weight.sub_(full_fc2_weight[-1:].clone())
                block.mlp.fc2.weight.copy_(full_fc2_weight[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeAdamW:
    """AdamW on relative coordinates with one implicit shift-gauge coordinate."""
=======
class GaugeAdamW:
    """AdamW on relative coordinates with an implicit final coordinate per column."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.exp_avg = torch.zeros(param.numel() + 1, device=param.device, dtype=param.dtype)
        self.exp_avg_sq = torch.zeros_like(self.exp_avg)
=======
        state_shape = (param.shape[0] + 1,) + tuple(param.shape[1:])
        self.exp_avg = torch.zeros(
            state_shape, device=param.device, dtype=param.dtype
        )
        self.exp_avg_sq = torch.zeros_like(self.exp_avg)
>>>>>>> REPLACE

<<<<<<< SEARCH
        grad = self.param.grad
        full_grad = torch.cat((grad, -grad.sum().reshape(1)))
=======
        grad = self.param.grad
        full_grad = torch.cat(
            (grad, -grad.sum(dim=0, keepdim=True)),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.param.add_(update[:-1] - update[-1], alpha=-self.lr)
=======
        self.param.add_(update[:-1] - update[-1:], alpha=-self.lr)
>>>>>>> REPLACE

<<<<<<< SEARCH
            total_sq.add_(param.grad.detach().float().sum().square())
=======
            total_sq.add_(
                param.grad.detach().float().sum(dim=0).square().sum()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.attn.proj_col_2 for block in model.blocks
    ] + [
        block.attn.proj_col for block in model.blocks
    ] + [
        block.mlp.fc2_col_2 for block in model.blocks
    ] + [
        block.mlp.fc2_col for block in model.blocks
    ] + [
        block.mlp.fc2_col_3 for block in model.blocks
    ] + [
        block.mlp.fc2_col_4 for block in model.blocks
    ] + [
        block.mlp.fc2_col_5 for block in model.blocks
    ] + [
        block.mlp.fc2_col_0 for block in model.blocks
    ] + [
        block.mlp.fc2_col_1 for block in model.blocks
    ] + [
        block.mlp.fc2_col_low for block in model.blocks
    ] + [
        block.mlp.fc2_col_mid for block in model.blocks
    ] + [
        block.mlp.fc2_col_4_abs for block in model.blocks
    ] + [
        block.mlp.fc2_col_5_abs for block in model.blocks
    ]
=======
    ] + [
        block.attn.proj_col_2 for block in model.blocks
    ] + [
        block.attn.proj_col for block in model.blocks
    ] + [
        block.mlp.fc2.weight for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Strictly avoid holdout leakage.
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
=======
        # Strictly avoid holdout leakage.
        for i, (ai, bi) in enumerate(zip(a.tolist(), b.tolist())):
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                a[i] = ai
                b[i] = bi
>>>>>>> REPLACE