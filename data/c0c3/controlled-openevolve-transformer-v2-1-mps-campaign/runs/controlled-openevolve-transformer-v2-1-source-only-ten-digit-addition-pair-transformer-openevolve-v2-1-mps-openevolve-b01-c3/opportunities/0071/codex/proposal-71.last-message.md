MECHANISM: Boundary-contrast extrapolation for sparse attention biases

HYPOTHESIS: A 1,271-parameter model will retain at least 99% accuracy because it preserves a head-specific bias at the fifteenth-farthest distance—the distinction lost by the failed 94.79% design—while deriving that contrast from the adjacent learned distance instead of allocating an independent parameter.

INTENDED_EDIT: Share the fifteen sparsest base biases across heads, but add the centered head contrast from the nearest fully learned distance to the first shared bin, reducing the current 1,274-parameter model by three parameters.

EVIDENCE: The 1,272-parameter design sharing fourteen far-distance bins achieved 99.3%, whereas fully sharing the fifteenth bin at 1,271 parameters fell to 94.79%; this motivates retaining its head distinction under a one-dimensional adjacency constraint.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
        self.far_rel_bias = nn.Parameter(torch.zeros(12))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 16))
        self.far_rel_bias = nn.Parameter(torch.zeros(15))
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_bias = torch.cat(
            (
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 12),
            ),
            dim=1,
        )
=======
        boundary = self.rel_bias[:, -1:]
        boundary_contrast = boundary - boundary.mean(dim=0, keepdim=True)
        far_bias = self.far_rel_bias.expand(self.n_head, 15)
        far_bias = torch.cat(
            (far_bias[:, :1] + boundary_contrast, far_bias[:, 1:]), dim=1
        )
        learned_bias = torch.cat((self.rel_bias, far_bias), dim=1)
>>>>>>> REPLACE