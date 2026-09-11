MECHANISM: Incremental companion-key coordinate alignment

HYPOTHESIS: Fixing the second tail coefficient of head 1’s first key channel will reduce the model from 990 to 989 parameters while retaining at least 99% accuracy, because fixing its adjacent leading coefficient achieved 99.97% without reducing the proven two-dimensional attention score rank.

INTENDED_EDIT: Omit the first two tail coefficients from head 1’s first key channel, reconstruct them as zeros, and project the virtual optimizer state onto this extended constraint after every update.

EVIDENCE: The immediately preceding result fixed one companion-channel tail coefficient and achieved 99.97% at 990 parameters, while fixing all five coefficients of head 1’s second key channel achieved 99.94%; extending the successful constraint by one adjacent coefficient is the smallest informative compression step.

<<<<<<< SEARCH
        # Head 1's second key channel has a fixed-zero tail. The leading
        # tail coefficient of its companion channel is also fixed at zero.
        self.key_tail = nn.Parameter(
            torch.empty(
                (n_head * self.score_dim - 1)
                * (reduced_width - self.score_dim)
                - 1
            )
        )
=======
        # Head 1's second key channel has a fixed-zero tail. The first two
        # tail coefficients of its companion channel are also fixed at zero.
        self.key_tail = nn.Parameter(
            torch.empty(
                (n_head * self.score_dim - 1)
                * (reduced_width - self.score_dim)
                - 2
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_key_tail[-2, 0].zero_()
            flat_key_tail[-1].zero_()
            self.key_tail.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 1:],
                    ]
                )
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(1),
                self.key_tail[learned_boundary:],
                self.key_tail.new_zeros(tail_width),
=======
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(2),
                self.key_tail[learned_boundary:],
                self.key_tail.new_zeros(tail_width),
>>>>>>> REPLACE

<<<<<<< SEARCH
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(1),
                stored_key_tail_grad[learned_boundary:],
                stored_key_tail_grad.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        detached_key_tail = key_tail_param.detach()
        canonical_key_tail = torch.cat(
            [
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(1),
                detached_key_tail[learned_boundary:],
                detached_key_tail.new_zeros(tail_width),
=======
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(2),
                stored_key_tail_grad[learned_boundary:],
                stored_key_tail_grad.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        detached_key_tail = key_tail_param.detach()
        canonical_key_tail = torch.cat(
            [
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(2),
                detached_key_tail[learned_boundary:],
                detached_key_tail.new_zeros(tail_width),
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_key_tail[-2, 0].zero_()
            flat_key_tail[-1].zero_()
            key_tail_param.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 1:],
                    ]
                )
            )
=======
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
>>>>>>> REPLACE