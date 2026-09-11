MECHANISM: Shared lexical-subspace nonlinear write

HYPOTHESIS: Constraining the MLP to write through the proven four-dimensional lexical subspace will reduce the verified model from 606 to 573 parameters while retaining at least 99% accuracy, because the residual already preserves routed positional information and the final logits consume only four learned lexical features.

INTENDED_EDIT: Replace the MLP’s independent seven-dimensional output projection and bias with four learned coordinates decoded through the centered shared token projection, initialize them by least-squares projection of the original initialized MLP, and optimize the reduced coordinates directly.

EVIDENCE: The 606-parameter model reached 99.89% with a rank-four lexical interface, whereas restricting positional amplitudes collapsed accuracy to 6.2% and pruning the attention-bias boundary collapsed it to 0.01%. This challenges the separate seven-dimensional nonlinear-write assumption while preserving the empirically load-bearing positional and routing mechanisms.

<<<<<<< SEARCH
    def gauge_fix_fc2(self) -> None:
        with torch.no_grad():
            weight = self.fc2.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.fc2.weight = nn.Parameter(
                (self.fc1_basis @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)
=======
    def gauge_fix_fc2(self) -> None:
        with torch.no_grad():
            weight = self.fc2.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.fc2.weight = nn.Parameter(
                (self.fc1_basis @ centered).clone()
            )

    def tie_output_to_lexical(self, token_proj: torch.Tensor) -> None:
        with torch.no_grad():
            weight = self.fc1_basis.T @ self.fc2.weight
            bias = self.fc1_basis.T @ self.fc2.bias
            lexical_proj = token_proj - token_proj.mean(
                dim=1, keepdim=True
            )
            gram = lexical_proj @ lexical_proj.T
            self.fc2.weight = nn.Parameter(
                torch.linalg.solve(
                    gram, lexical_proj @ weight
                ).clone()
            )
            self.fc2.bias = nn.Parameter(
                torch.linalg.solve(
                    gram, lexical_proj @ bias
                ).clone()
            )

    def forward(
        self, x: torch.Tensor, token_proj: torch.Tensor
    ) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        latent_output = F.linear(
            F.gelu(hidden), self.fc2.weight, self.fc2.bias
        )
        lexical_proj = token_proj - token_proj.mean(
            dim=1, keepdim=True
        )
        output = latent_output @ lexical_proj
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(
        self, x: torch.Tensor, token_proj: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), token_proj)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
        self.token_emb = None
        self.lm_head = None
=======
        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
        for block in self.blocks:
            block.mlp.tie_output_to_lexical(self.token_proj)
        self.token_emb = None
        self.lm_head = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in self.blocks:
            x = blk(x)
=======
        for blk in self.blocks:
            x = blk(x, token_proj)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # AdamW's coordinate-wise second moments are not invariant to an
    # orthogonal reparameterization. Update the compressed fc2 biases
    # separately using virtual moments in their original eight coordinates.
    fc2_bias_params = [block.mlp.fc2.bias for block in model.blocks]
    fc2_bias_ids = {id(param) for param in fc2_bias_params}
    token_phase_id = id(model.token_phase)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for param in model.parameters()
                    if id(param) not in fc2_bias_ids
                    and id(param) != token_phase_id
                ],
            },
            {
                "params": [model.token_phase],
                "weight_decay": 0.0,
            },
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
=======
    token_phase_id = id(model.token_phase)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for param in model.parameters()
                    if id(param) != token_phase_id
                ],
            },
            {
                "params": [model.token_phase],
                "weight_decay": 0.0,
            },
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        for bias in fc2_bias_params:
            bias.grad = None
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
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
=======
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
>>>>>>> REPLACE