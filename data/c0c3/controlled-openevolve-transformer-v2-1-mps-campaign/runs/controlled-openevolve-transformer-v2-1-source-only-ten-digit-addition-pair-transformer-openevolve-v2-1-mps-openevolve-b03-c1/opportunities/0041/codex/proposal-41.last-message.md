MECHANISM: Gauge-fixed rank-six lexical chart

HYPOTHESIS: Removing the 36-dimensional factorization gauge from the successful rank-six tied token table will reduce learned parameters from 1,259 to 1,223 while retaining at least 99% accuracy, because the reconstructed token table has unchanged rank-six expressivity and is initialized identically.

INTENDED_EDIT: Select the best-conditioned six feature columns as a fixed coordinate chart, absorb their invertible projection into the token codes, learn only the remaining 6-by-2 projection, and reconstruct the full tied token table during forward passes.

EVIDENCE: The rank-six tied lexical bottleneck achieved 99.87% accuracy at 1,259 parameters; its two learned factors contain an exact 6-by-6 change-of-basis redundancy, so gauge-fixing that redundancy is a more conservative next reduction than lowering lexical rank.

<<<<<<< SEARCH
        # Challenge the dense-lexicon assumption shared by the prior designs.
        # A learned rank-six table supplies token-specific latent codes, while
        # a shared learned projection maps those codes into the residual stream.
        # Reusing both factors for the output preserves weight tying.
        token_rank = 6
        with torch.no_grad():
            token_left, token_singular, token_right = torch.linalg.svd(
                self.token_emb.weight, full_matrices=False
            )
            token_scale = token_singular[:token_rank].sqrt()
            token_code = token_left[:, :token_rank] * token_scale
            token_proj = token_scale.unsqueeze(1) * token_right[:token_rank]
        self.token_code = nn.Parameter(token_code.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
        self.token_emb = None
        self.lm_head = None
=======
        # A rank-six product is invariant under an invertible change of latent
        # basis. Fix that 36-dimensional gauge by choosing the best-conditioned
        # six feature columns as an identity pivot and learning only the two
        # remaining columns.
        token_rank = 6
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        position = F.embedding(pos, self.pos_code) @ pos_proj
        token = F.embedding(idx, self.token_code) @ self.token_proj
        x = token + position
=======
        position = F.embedding(pos, self.pos_code) @ pos_proj
        token_proj = torch.cat(
            (self.token_pivot, self.token_proj), dim=1
        )[:, self.token_inverse_order]
        token = F.embedding(idx, self.token_code) @ token_proj
        x = token + position
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        token_features = x @ self.token_proj.T
        logits = F.linear(token_features, self.token_code)
=======
        x = self.ln_f(x)
        token_features = x @ token_proj.T
        logits = F.linear(token_features, self.token_code)
>>>>>>> REPLACE