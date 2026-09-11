MECHANISM: Fourth attention-output column gauge with ambient AdamW

HYPOTHESIS: A 1,529-parameter model will retain at least 99% accuracy because the verified three-column attention-projection gauge achieved 99.93%, while a fourth column’s common-output component is likewise erased by downstream LayerNorm and its full AdamW dynamics remain preserved.

INTENDED_EDIT: Reproduce the qualified three-column attention-output gauge and extend it to four columns, including initialization, gradients, clipping contribution, optimizer moments, and ambient-coordinate updates.

EVIDENCE: Reference Design 3 achieved 99.93% accuracy at 1,530 parameters with three gauged attention-projection columns; the successful one-to-two and two-to-three reductions motivate testing the next exact symmetry along this axis.

<<<<<<< SEARCH
    """Linear projection with bias and one weight-column output gauge removed."""
=======
    """Linear projection with bias and four weight-column output gauges removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_prefix = nn.Parameter(torch.empty(out_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
=======
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_prefix.copy_(
            raw_weight[:-1, 0] - raw_weight[-1, 0]
        )
        self.weight_rest.copy_(raw_weight[:, 1:])
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_weight_prefix = torch.cat(
            (self.weight_prefix, self.weight_prefix.new_zeros(1))
        )
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_weight_prefix.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (full_weight_prefix.unsqueeze(1), self.weight_rest), dim=1
        )
=======
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_prefix.copy_(
                    raw_weight[:-1, 0] - raw_weight[-1, 0]
                )
                module.weight_rest.copy_(raw_weight[:, 1:])
=======
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.append(blk.attn.proj.weight_prefix)
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, four attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.proj.weight_prefix)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            full_gauge_grads.append(
                blk.attn.proj.full_weight_prefix.grad.detach()
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
=======
        for blk in model.blocks:
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
>>>>>>> REPLACE