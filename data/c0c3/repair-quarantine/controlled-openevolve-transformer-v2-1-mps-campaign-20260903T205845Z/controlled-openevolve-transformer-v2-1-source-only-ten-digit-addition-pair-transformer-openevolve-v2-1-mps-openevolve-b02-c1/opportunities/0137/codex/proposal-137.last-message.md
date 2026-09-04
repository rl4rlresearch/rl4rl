MECHANISM: Extended companion-key sparsification

HYPOTHESIS: Fixing one additional adjacent coefficient in head 1’s already-sparse companion key tail will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend the fixed-zero prefix of head 1’s companion key-tail channel from two coefficients to three, updating initialization, reconstruction, gradient lifting, and optimizer projection consistently.

EVIDENCE: The verified 984-parameter model reached 99.75% while already fixing this channel’s first two coefficients and the entire neighboring key tail; unlike the failed final-LayerNorm tie and far-distance bias extension, this incrementally extends a successful sparsity pattern.

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
        # Head 1's second key channel has a fixed-zero tail. The first three
        # tail coefficients of its companion channel are also fixed at zero.
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
            flat_key_tail[-2, :3].zero_()
            flat_key_tail[-1].zero_()
            self.key_tail.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 3:],
                    ]
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(2),
                self.key_tail[learned_boundary:],
                self.key_tail.new_zeros(tail_width),
=======
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(3),
                self.key_tail[learned_boundary:],
                self.key_tail.new_zeros(tail_width),
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
        detached_key_tail = key_tail_param.detach()
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
        canonical_key_tail_grad = torch.cat(
            [
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(3),
                stored_key_tail_grad[learned_boundary:],
                stored_key_tail_grad.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        detached_key_tail = key_tail_param.detach()
        canonical_key_tail = torch.cat(
            [
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(3),
                detached_key_tail[learned_boundary:],
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
            flat_key_tail[-2, :3].zero_()
            flat_key_tail[-1].zero_()
            key_tail_param.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 3:],
                    ]
                )
            )
>>>>>>> REPLACE