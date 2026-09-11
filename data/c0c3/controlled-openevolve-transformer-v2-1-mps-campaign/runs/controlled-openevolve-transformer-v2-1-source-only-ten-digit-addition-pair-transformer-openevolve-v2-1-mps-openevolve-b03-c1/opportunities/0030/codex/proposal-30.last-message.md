MECHANISM: Fixed harmonic positional coordinates with learned dense readout

HYPOTHESIS: Replacing the adaptive rank-four positional lookup with two fixed Fourier harmonics will remove `4 * max_seq_len - 8` learned parameters while retaining at least 99% accuracy, because learned full query/key projections can turn harmonic coordinates into sum- and difference-of-position alignment features, while the relative-bias table continues to provide exact distance cues.

INTENDED_EDIT: Freeze absolute position representation to generic two-frequency sine/cosine coordinates, restore a fully learned 4-by-8 positional readout, and remove the positional-factor reconstruction and its fixed pivots.

EVIDENCE: Rank-four absolute codes plus relative-distance attention reached 99.97%, showing four positional coordinates suffice; the shared-key collapse to 36.45% argues for preserving the full learned query/key machinery that can interpret the fixed harmonic coordinates.

<<<<<<< SEARCH
        # Preserve the baseline RNG stream by initializing the original table,
        # then replace it with its best rank-four factorization. Orthogonal
        # latent-basis rotations fix three redundant projection entries at zero
        # without changing the initialized positional table or factor geometry.
=======
        # Preserve the baseline RNG stream by initializing the original table.
        # The transient factorization below keeps initialization behavior aligned
        # with the prior design before fixed harmonic coordinates replace it.
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_code = nn.Parameter(pos_code)
        # Fix two scale-gauge coordinates at their initialized values. Changes
        # to either pivot are representable by inversely rescaling the matching
        # positional-code column and projection row.
        self.register_buffer("pos_proj_row2_pivot", pos_proj[2, 1:2].clone())
        self.register_buffer("pos_proj_pivot", pos_proj[3, :1].clone())
        self.pos_proj = nn.Parameter(
            torch.cat(
                (
                    pos_proj[0, 3:],
                    pos_proj[1, 2:],
                    pos_proj[2, 2:],
                    pos_proj[3, 1:],
                )
            )
        )
        self.pos_emb = None
=======
        # Generic fixed harmonics expose both position sums and differences to
        # the learned query/key bilinear forms without a learned code per slot.
        positions = torch.arange(
            cfg.max_seq_len,
            device=self.pos_emb.weight.device,
            dtype=self.pos_emb.weight.dtype,
        )
        phase = 2.0 * math.pi * positions / cfg.max_seq_len
        pos_code = torch.stack(
            (
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(2.0 * phase),
                torch.cos(2.0 * phase),
            ),
            dim=-1,
        ) / math.sqrt(2.0)
        self.register_buffer("pos_code", pos_code)
        self.pos_proj = nn.Parameter(self.pos_emb.weight[:4].detach().clone())
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_proj_row = torch.cat(
            (self.pos_proj.new_zeros(3), self.pos_proj[: self.cfg.d_model - 3])
        )
        second_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(2),
                self.pos_proj[
                    self.cfg.d_model - 3 : 2 * self.cfg.d_model - 5
                ],
            )
        )
        third_proj_row = torch.cat(
            (
                self.pos_proj.new_zeros(1),
                self.pos_proj_row2_pivot,
                self.pos_proj[
                    2 * self.cfg.d_model - 5 : 3 * self.cfg.d_model - 7
                ],
            )
        )
        fourth_proj_row = torch.cat(
            (
                self.pos_proj_pivot,
                self.pos_proj[3 * self.cfg.d_model - 7 :],
            )
        )
        pos_proj = torch.cat(
            (
                first_proj_row.unsqueeze(0),
                second_proj_row.unsqueeze(0),
                third_proj_row.unsqueeze(0),
                fourth_proj_row.unsqueeze(0),
            ),
            dim=0,
        )
        position = F.embedding(pos, self.pos_code) @ pos_proj
=======
        position = F.embedding(pos, self.pos_code) @ self.pos_proj
>>>>>>> REPLACE