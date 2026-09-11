MECHANISM: Orthogonal gauge fixing of the low-rank positional factorization

HYPOTHESIS: Removing one rotationally redundant scalar from the rank-four positional factors will reduce parameters from 1,586 to 1,585 while retaining at least 99% accuracy, because an orthogonal basis change preserves the initialized positional table and factor-space geometry while fixing one projection entry to zero.

INTENDED_EDIT: Rotate the first two positional latent dimensions so `pos_proj[0, 0]` is zero, omit that scalar from the learned parameter, and reconstruct the fixed zero during inference and training.

EVIDENCE: The rank-four positional/relative-attention model achieved 99.97%, and exact relative-bias gauge fixing retained 99.96%; this tests another exact redundancy within the successful low-rank positional parameterization.

<<<<<<< SEARCH
        # Preserve the baseline RNG stream by initializing the original table,
        # then replace it with its best rank-four factorization.
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(
                self.pos_emb.weight, full_matrices=False
            )
            scale = singular[:4].sqrt()
            pos_code = left[:, :4] * scale
            pos_proj = scale.unsqueeze(1) * right[:4]
        self.pos_code = nn.Parameter(pos_code)
        self.pos_proj = nn.Parameter(pos_proj)
        self.pos_emb = None
=======
        # Preserve the baseline RNG stream by initializing the original table,
        # then replace it with its best rank-four factorization. An orthogonal
        # latent-basis rotation fixes one redundant projection entry at zero
        # without changing the initialized positional table.
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(
                self.pos_emb.weight, full_matrices=False
            )
            scale = singular[:4].sqrt()
            pos_code = left[:, :4] * scale
            pos_proj = scale.unsqueeze(1) * right[:4]

            a = pos_proj[0, 0]
            b = pos_proj[1, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[0, 0] = b / radius
            rotation[0, 1] = -a / radius
            rotation[1, 0] = a / radius
            rotation[1, 1] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

        self.pos_code = nn.Parameter(pos_code)
        self.pos_proj = nn.Parameter(
            torch.cat((pos_proj[0, 1:], pos_proj[1:].reshape(-1)))
        )
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        position = F.embedding(pos, self.pos_code) @ self.pos_proj
        x = self.token_emb(idx) + position
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        first_proj_row = torch.cat(
            (self.pos_proj.new_zeros(1), self.pos_proj[: self.cfg.d_model - 1])
        )
        pos_proj = torch.cat(
            (
                first_proj_row.unsqueeze(0),
                self.pos_proj[self.cfg.d_model - 1 :].view(3, self.cfg.d_model),
            ),
            dim=0,
        )
        position = F.embedding(pos, self.pos_code) @ pos_proj
        x = self.token_emb(idx) + position
>>>>>>> REPLACE