MECHANISM: Learned scalar-phase harmonic token representation

HYPOTHESIS: Constraining each token’s learned rank-four code to a trainable radius and phase will reduce the model from 850 to 638 parameters while retaining at least 99% accuracy, because it preserves four ambient lexical channels while replacing four independent coordinates per token with two learned semantic coordinates.

INTENDED_EDIT: Generate tied input/output token codes by harmonically lifting learned per-token phases and radii into four dimensions, restore a full learned token-to-residual projection, and exempt periodic phase coordinates from weight decay.

EVIDENCE: Rank four achieved 99.98% while rank three collapsed to 79.28%, showing that four ambient lexical channels are load-bearing. The successful 850-parameter model does not establish that every token needs four independently learned coordinates, so a two-coordinate harmonic manifold cleanly tests that shared assumption without narrowing the proven four-dimensional interface.

<<<<<<< SEARCH
        # A rank-four product is invariant under an invertible change of latent
        # basis. Fix that 16-dimensional gauge by choosing the best-conditioned
        # four feature columns as an identity pivot and learning only the four
        # remaining columns.
        with torch.no_grad():
            token_left, token_singular, token_right = torch.linalg.svd(
                self.token_emb.weight, full_matrices=False
            )
            token_scale = token_singular[:token_rank].sqrt()
            token_code = token_left[:, :token_rank] * token_scale
            token_proj = token_scale.unsqueeze(1) * token_right[:token_rank]

            candidates = torch.combinations(
                torch.arange(cfg.d_model, device=token_proj.device),
                r=token_rank,
            )
            candidate_matrices = token_proj[:, candidates].permute(1, 0, 2)
            pivot_columns = candidates[
                torch.linalg.det(candidate_matrices).abs().argmax()
            ]
            all_columns = torch.arange(cfg.d_model, device=token_proj.device)
            tail_columns = all_columns[
                (all_columns[:, None] != pivot_columns[None, :]).all(dim=1)
            ]
            column_order = torch.cat((pivot_columns, tail_columns))
            pivot = token_proj[:, pivot_columns]

            token_code = token_code @ pivot
            token_tail = torch.linalg.solve(
                pivot, token_proj[:, tail_columns]
            )

        self.token_code = nn.Parameter(token_code.clone())
        self.token_proj = nn.Parameter(token_tail.clone())
        self.register_buffer(
            "token_pivot",
            torch.eye(
                token_rank,
                device=token_proj.device,
                dtype=token_proj.dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "token_inverse_order", torch.argsort(column_order)
        )
        self.token_emb = None
        self.lm_head = None
=======
        # Preserve the proven four-dimensional lexical interface, but challenge
        # the assumption that every token needs four independent coordinates.
        # A learned scalar phase and radius are lifted onto a trigonometric
        # moment curve whose ambient span remains rank four.
        with torch.no_grad():
            token_left, token_singular, token_right = torch.linalg.svd(
                self.token_emb.weight, full_matrices=False
            )
            token_scale = token_singular[:token_rank].sqrt()
            initialized_token_code = (
                token_left[:, :token_rank] * token_scale
            )
            token_proj = (
                token_scale.unsqueeze(1) * token_right[:token_rank]
            )
            token_radius = torch.linalg.vector_norm(
                initialized_token_code, dim=1
            )
            token_phase = torch.atan2(
                initialized_token_code[:, 0],
                initialized_token_code[:, 1],
            )

        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
        self.token_emb = None
        self.lm_head = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_proj = torch.cat(
            (self.token_pivot, self.token_proj), dim=1
        )[:, self.token_inverse_order]
        token = F.embedding(idx, self.token_code) @ token_proj
=======
        token_code = self.token_radius.unsqueeze(1) * torch.stack(
            (
                torch.sin(self.token_phase),
                torch.cos(self.token_phase),
                torch.sin(2.0 * self.token_phase),
                torch.cos(2.0 * self.token_phase),
            ),
            dim=1,
        ) / math.sqrt(2.0)
        token_proj = self.token_proj
        token = F.embedding(idx, token_code) @ token_proj
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_features = x @ token_proj.T + self.ln_f_token_bias
        logits = F.linear(token_features, self.token_code)
=======
        token_features = x @ token_proj.T + self.ln_f_token_bias
        logits = F.linear(token_features, token_code)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE