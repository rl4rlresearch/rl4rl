MECHANISM: Head-specific learned radial relative-position routing

HYPOTHESIS: Replacing independent distance-lookup biases with three trainable Gaussian routing bands per head will substantially reduce parameters while retaining at least 99% accuracy, because successful far-distance sharing indicates that routing profiles are structured, while keeping separate kernels and value maps preserves the head specialization whose removal failed.

INTENDED_EDIT: Replace both relative-bias lookup tables with 18 learned amplitude/center/log-width parameters that generate continuous head-specific bias profiles, and optimize them directly with AdamW rather than the lookup-table quotient update.

EVIDENCE: Extending head 0’s tied far-distance region from two through five positions retained at least 99.89% accuracy, while sharing value maps across heads collapsed accuracy to 15.71%; this supports compressing positional routing into independent low-dimensional kernels without sharing the heads’ learned content representations.

<<<<<<< SEARCH
        # Head 0 ties the five farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 5)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Each head learns three localized relative-position routing bands.
        # Their amplitudes begin at zero, preserving unbiased initial
        # attention while the centers and widths remain fully trainable.
        centers = torch.linspace(
            0.2 * (max_seq_len - 1),
            0.8 * (max_seq_len - 1),
            3,
        )
        components = torch.zeros(n_head, 3, 3)
        components[:, :, 1] = centers
        components[:, :, 2] = math.log(max(1.0, max_seq_len / 6.0))
        self.rel_bias = nn.Parameter(components)
>>>>>>> REPLACE

<<<<<<< SEARCH
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        rel_bias = torch.stack(
            [
                F.pad(self.rel_bias[0], (0, 5)),
                torch.cat(
                    [
                        self.rel_bias[1][:-1],
                        0.75 * self.rel_bias[1][-2:-1]
                        + 0.25 * self.rel_bias[1][-1:],
                        0.375 * self.rel_bias[1][-2:-1]
                        + 0.625 * self.rel_bias[1][-1:],
                        self.rel_bias[1][-1:].expand(4),
                        self.rel_bias[1].new_zeros(3),
                    ]
                ),
            ]
        )
        att = att + rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
=======
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        distance_axis = torch.arange(
            self.mask.size(0),
            device=x.device,
            dtype=att.dtype,
        ).view(1, 1, -1)
        components = self.rel_bias.to(dtype=att.dtype)
        amplitude = components[..., 0:1]
        center = components[..., 1:2]
        width = components[..., 2:3].exp().clamp(
            min=0.5, max=float(self.mask.size(0))
        )
        radial_features = torch.exp(
            -0.5 * ((distance_axis - center) / width).square()
        )
        rel_bias = (amplitude * radial_features).sum(dim=1)
        att = att + rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (head_bias, 0)
            for block in model.blocks
            for head_bias in block.attn.rel_bias
        ]
=======
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ]
>>>>>>> REPLACE