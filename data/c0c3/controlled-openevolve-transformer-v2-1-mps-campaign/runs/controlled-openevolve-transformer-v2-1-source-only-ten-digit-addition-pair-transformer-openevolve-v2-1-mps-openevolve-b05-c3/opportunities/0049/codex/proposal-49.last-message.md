MECHANISM: Final-MLP common-output-bias quotient

HYPOTHESIS: Gauge-fixing the common shift of the final MLP output bias will produce 1,593 learned parameters while retaining at least 99% accuracy, because this shift adds only a channel-uniform residual offset removed by the final LayerNorm.

INTENDED_EDIT: Represent each `fc2` bias with seven learned coordinates and a fixed final zero, while using virtual full-space AdamW moments, weight decay, and gradient clipping for the omitted coordinate.

EVIDENCE: The two-column attention-output quotient reached 99.95% at 1,594 parameters, demonstrating that common-output shifts can be removed successfully; the third attention-column quotient failed at 71.64%, motivating an orthogonal common-output gauge immediately before the final LayerNorm instead of another attention input-column reduction.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    key_gauges: List[
=======
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    output_bias_gauges: List[torch.nn.Parameter],
    key_gauges: List[
>>>>>>> REPLACE

<<<<<<< SEARCH
        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())

        for key_parameter, ln_scale, d_model in key_gauges:
=======
        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())

        if any(
            parameter is output_bias
            for output_bias in output_bias_gauges
        ):
            total_sq.add_(grad.sum().square())

        for key_parameter, ln_scale, d_model in key_gauges:
>>>>>>> REPLACE

<<<<<<< SEARCH
    position_parameter = model.pos_emb.weight
    key_gauge_modules = [
=======
    position_parameter = model.pos_emb.weight
    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
    key_gauge_modules = [
>>>>>>> REPLACE

<<<<<<< SEARCH
    excluded = {id(position_parameter)}
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
=======
    excluded = {id(position_parameter)}
    excluded.update(
        id(parameter) for parameter in output_bias_gauge_parameters
    )
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
>>>>>>> REPLACE

<<<<<<< SEARCH
    position_optimizer = GaugeAdamW(
        position_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizers = [
=======
    position_optimizer = GaugeAdamW(
        position_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    output_bias_optimizers = [
        GaugeAdamW(
            parameter,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for parameter in output_bias_gauge_parameters
    ]
    key_optimizers = [
>>>>>>> REPLACE

<<<<<<< SEARCH
    all_optimizers = [
        optimizer,
        position_optimizer,
        *key_optimizers,
        *projection_optimizers,
        *shared_optimizers,
    ]
=======
    all_optimizers = [
        optimizer,
        position_optimizer,
        *output_bias_optimizers,
        *key_optimizers,
        *projection_optimizers,
        *shared_optimizers,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_gauges(
                model,
                position_parameter,
                key_gauges,
                projection_gauges,
=======
            clip_grad_norm_with_gauges(
                model,
                position_parameter,
                output_bias_gauge_parameters,
                key_gauges,
                projection_gauges,
>>>>>>> REPLACE

<<<<<<< SEARCH
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        position_optimizer.step()
        optimizer.step()
=======
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        for output_bias_optimizer in output_bias_optimizers:
            output_bias_optimizer.step()
        position_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE