MECHANISM: Canonical-basis harmonic position encoding

HYPOTHESIS: Replacing the 28-parameter dense positional readout with four learned harmonic amplitudes will reduce the model from 1,146 to 1,122 parameters while retaining at least 99% accuracy, because the independent learned query/key projections can mix a fixed orthogonal positional basis without requiring a separately learned direction for every harmonic-coordinate pair.

INTENDED_EDIT: Represent the four fixed harmonics along deterministic orthogonal zero-mean residual directions, learning only one amplitude per harmonic while preserving the initialized positional magnitude and all lexical, MLP, and head-specific routing capacity.

EVIDENCE: The 1,146-parameter design reaches 99.96% with fixed harmonic coordinates, showing that slot-specific learned position codes are unnecessary; meanwhile, `d_ff=10` and rank-five lexical compression failed, so this patch preserves those load-bearing capacities and tests the distinct assumption that dense learned positional orientation is necessary.

<<<<<<< SEARCH
        self.register_buffer("pos_code", pos_code)
        # A position-dependent shift shared by all model coordinates is removed
        # by every downstream LayerNorm. Center each readout row to fix this
        # exact gauge, then omit its final coordinate.
        pos_proj = self.pos_emb.weight[:4].detach().clone()
        pos_proj = pos_proj - pos_proj.mean(dim=1, keepdim=True)
        self.pos_proj = nn.Parameter(pos_proj[:, :-1].clone())
        self.pos_emb = None
=======
        self.register_buffer("pos_code", pos_code)

        # Give each harmonic a fixed orthogonal direction in the observable
        # zero-mean residual subspace. Learned query/key maps can perform the
        # required mixing, so retain only one trainable amplitude per harmonic.
        pos_basis = torch.eye(
            cfg.d_model,
            device=self.pos_emb.weight.device,
            dtype=self.pos_emb.weight.dtype,
        )[:, :4]
        pos_basis = pos_basis - pos_basis.mean(dim=0, keepdim=True)
        pos_basis, _ = torch.linalg.qr(pos_basis, mode="reduced")
        self.register_buffer(
            "pos_basis", pos_basis.T.contiguous(), persistent=False
        )

        initialized_pos_rows = self.pos_emb.weight[:4].detach().clone()
        initialized_pos_rows = initialized_pos_rows - initialized_pos_rows.mean(
            dim=1, keepdim=True
        )
        self.pos_scale = nn.Parameter(
            torch.linalg.vector_norm(initialized_pos_rows, dim=1)
        )
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_proj = torch.cat(
            (self.pos_proj, -self.pos_proj.sum(dim=1, keepdim=True)), dim=1
        )
        position = F.embedding(pos, self.pos_code) @ pos_proj
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_proj = self.pos_scale.unsqueeze(1) * self.pos_basis
        position = F.embedding(pos, self.pos_code) @ pos_proj
>>>>>>> REPLACE