MECHANISM: Optimizer-equivariant LayerNorm bias gauge fixing

HYPOTHESIS: Compressing `fc2.bias` to its seven observable zero-mean coordinates while emulating the original eight-coordinate AdamW updates will reduce the model from 1,119 to 1,118 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Reconstruct the full zero-mean MLP output bias from seven learned coordinates and train those coordinates using projected updates from virtual eight-dimensional Adam moments.

EVIDENCE: The current model achieved 99.53%, while the earlier orthonormal `fc2.bias` compression collapsed to 5.8%; because AdamW is not invariant to orthogonal reparameterization, preserving its original coordinate-wise moments directly tests whether optimization—not representational capacity—caused that failure.

<<<<<<< SEARCH
        self.register_buffer("fc1_basis", basis.T, persistent=False)
=======
        self.register_buffer("fc1_basis", basis.T, persistent=False)

        # The downstream LayerNorm removes the all-ones component of this
        # bias. Retain only its seven observable zero-mean coordinates.
        self.fc2.bias = nn.Parameter(self.fc2.bias[:-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        output = F.linear(F.gelu(hidden), fc2_weight, self.fc2.bias)
=======
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    # AdamW's coordinate-wise second moments are not invariant to an
    # orthogonal reparameterization. Update the compressed fc2 biases
    # separately using virtual moments in their original eight coordinates.
    fc2_bias_params = [block.mlp.fc2.bias for block in model.blocks]
    fc2_bias_ids = {id(param) for param in fc2_bias_params}
    optimizer = torch.optim.AdamW(
        [
            param
            for param in model.parameters()
            if id(param) not in fc2_bias_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    fc2_bias_states = [
        {
            "exp_avg": bias.new_zeros(block.mlp.fc1_basis.size(1)),
            "exp_avg_sq": bias.new_zeros(block.mlp.fc1_basis.size(1)),
        }
        for block, bias in zip(model.blocks, fc2_bias_params)
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
=======
        optimizer.zero_grad(set_to_none=True)
        for bias in fc2_bias_params:
            bias.grad = None
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()
=======
        with torch.no_grad():
            adam_step = step + 1
            beta1, beta2 = 0.9, 0.999
            for block, bias, state in zip(
                model.blocks, fc2_bias_params, fc2_bias_states
            ):
                full_grad = block.mlp.fc1_basis.T @ bias.grad
                state["exp_avg"].mul_(beta1).add_(
                    full_grad, alpha=1.0 - beta1
                )
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )
                full_update = state["exp_avg"] / (
                    1.0 - beta1 ** adam_step
                )
                denom = (
                    state["exp_avg_sq"] / (1.0 - beta2 ** adam_step)
                ).sqrt().add_(1e-8)
                reduced_update = block.mlp.fc1_basis @ (
                    full_update / denom
                )
                bias.mul_(1.0 - lr_now * train_cfg.weight_decay)
                bias.add_(reduced_update, alpha=-lr_now)
        optimizer.step()
>>>>>>> REPLACE