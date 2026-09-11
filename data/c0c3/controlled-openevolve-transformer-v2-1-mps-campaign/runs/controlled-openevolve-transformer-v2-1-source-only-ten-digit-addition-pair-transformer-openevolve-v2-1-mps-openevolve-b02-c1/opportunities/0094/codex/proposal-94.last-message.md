MECHANISM: Selective companion-key coordinate alignment

HYPOTHESIS: Fixing the fourth tail coefficient of head 1’s first key channel will reduce the model from 989 to 988 parameters while retaining at least 99% accuracy, because the first two coefficients tolerate zero constraints, the third is specifically load-bearing, and this leaves the terminal coefficient aligned with the reference-sensitive seventh MLP input coordinate learned independently.

INTENDED_EDIT: Omit the fourth tail coefficient from head 1’s first key channel, reconstruct it as zero alongside the existing constraints, and project the virtual optimizer state onto the same constraint after every update.

EVIDENCE: Fixing the first two companion-channel tail coefficients achieved 99.98% at 989 parameters, while additionally fixing the third collapsed to 0%; this indicates coefficient-specific sensitivity and motivates testing the next nonterminal coefficient without altering the critical third or terminal fifth coefficient.

<<<<<<< SEARCH
        # Head 1's second key channel has a fixed-zero tail. The first two
        # tail coefficients of its companion channel are also fixed at zero.
        self.key_tail = nn.Parameter(
            torch.empty(
                (n_head * self.score_dim - 1)
                * (reduced_width - self.score_dim)
                - 2
            )
        )
=======
        # Head 1's second key channel has a fixed-zero tail. The first two
        # and fourth tail coefficients of its companion channel are fixed.
        self.key_tail = nn.Parameter(
            torch.empty(
                (n_head * self.score_dim - 1)
                * (reduced_width - self.score_dim)
                - 3
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_key_tail[-2, :2].zero_()
            flat_key_tail[-1].zero_()
            self.key_tail.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 2:],
                    ]
                )
            )
=======
            flat_key_tail[-2, :2].zero_()
            flat_key_tail[-2, 3].zero_()
            flat_key_tail[-1].zero_()
            self.key_tail.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 2:3],
                        flat_key_tail[-2, 4:],
                    ]
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_tail = torch.cat(
            [
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(2),
                self.key_tail[learned_boundary:],
                self.key_tail.new_zeros(tail_width),
            ],
            dim=0,
        ).view(
=======
        key_tail = torch.cat(
            [
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(2),
                self.key_tail[learned_boundary:learned_boundary + 1],
                self.key_tail.new_zeros(1),
                self.key_tail[learned_boundary + 1:],
                self.key_tail.new_zeros(tail_width),
            ],
            dim=0,
        ).view(
>>>>>>> REPLACE

<<<<<<< SEARCH
        canonical_key_tail_grad = torch.cat(
            [
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(2),
                stored_key_tail_grad[learned_boundary:],
                stored_key_tail_grad.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
=======
        canonical_key_tail_grad = torch.cat(
            [
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(2),
                stored_key_tail_grad[
                    learned_boundary:learned_boundary + 1
                ],
                stored_key_tail_grad.new_zeros(1),
                stored_key_tail_grad[learned_boundary + 1:],
                stored_key_tail_grad.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
>>>>>>> REPLACE

<<<<<<< SEARCH
        canonical_key_tail = torch.cat(
            [
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(2),
                detached_key_tail[learned_boundary:],
                detached_key_tail.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
=======
        canonical_key_tail = torch.cat(
            [
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(2),
                detached_key_tail[
                    learned_boundary:learned_boundary + 1
                ],
                detached_key_tail.new_zeros(1),
                detached_key_tail[learned_boundary + 1:],
                detached_key_tail.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_key_tail[-2, :2].zero_()
            flat_key_tail[-1].zero_()
            key_tail_param.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 2:],
                    ]
                )
            )
=======
            flat_key_tail[-2, :2].zero_()
            flat_key_tail[-2, 3].zero_()
            flat_key_tail[-1].zero_()
            key_tail_param.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 2:3],
                        flat_key_tail[-2, 4:],
                    ]
                )
            )
>>>>>>> REPLACE