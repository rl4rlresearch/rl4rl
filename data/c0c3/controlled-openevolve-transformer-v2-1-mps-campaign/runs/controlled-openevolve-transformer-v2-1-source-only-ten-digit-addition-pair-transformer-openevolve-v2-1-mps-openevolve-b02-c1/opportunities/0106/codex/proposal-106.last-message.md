MECHANISM: Shared load-bearing companion-key pair

HYPOTHESIS: Tying head 1’s third and fourth companion-key tail coefficients will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because both coefficients require learned nonzero contributions and correspond to residual coordinates already successfully paired by the MLP.

INTENDED_EDIT: Replace the two independently stored companion-key coefficients with one shared coefficient, initialize it from their mean, split its gradient across both virtual coordinates, and reproject the virtual key factor after every optimizer update.

EVIDENCE: Fixing the third coefficient caused complete failure and fixing the fourth reached only 92.14%, showing both are load-bearing; meanwhile, disjoint MLP input pairing achieved 99.93%, specifically supporting shared treatment of their corresponding residual coordinates without deleting their learned contribution.

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
        # tail coefficients of its companion channel are fixed at zero, and
        # its third and fourth coefficients share one learned value.
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
            shared_tail = flat_key_tail[-2, 2:4].mean()
            flat_key_tail[-2, 2:4].copy_(shared_tail.expand(2))
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
        tail_width = reduced_x.shape[-1] - self.score_dim
        learned_boundary = (
            (self.n_head * self.score_dim - 2) * tail_width
        )
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
        tail_width = reduced_x.shape[-1] - self.score_dim
        learned_boundary = (
            (self.n_head * self.score_dim - 2) * tail_width
        )
        shared_key_tail = self.key_tail[
            learned_boundary : learned_boundary + 1
        ]
        key_tail = torch.cat(
            [
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(2),
                shared_key_tail.expand(2),
                self.key_tail[learned_boundary + 1 :],
                self.key_tail.new_zeros(tail_width),
            ],
            dim=0,
        ).view(
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_boundary = (
            (n_head * head_dim - 2) * tail_width
        )
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
        learned_boundary = (
            (n_head * head_dim - 2) * tail_width
        )
        shared_tail_grad = 0.5 * stored_key_tail_grad[
            learned_boundary : learned_boundary + 1
        ]
        canonical_key_tail_grad = torch.cat(
            [
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(2),
                shared_tail_grad.expand(2),
                stored_key_tail_grad[learned_boundary + 1 :],
                stored_key_tail_grad.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        detached_key_tail = key_tail_param.detach()
        shared_key_tail = detached_key_tail[
            learned_boundary : learned_boundary + 1
        ]
        canonical_key_tail = torch.cat(
            [
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(2),
                shared_key_tail.expand(2),
                detached_key_tail[learned_boundary + 1 :],
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
            shared_tail = flat_key_tail[-2, 2:4].mean()
            flat_key_tail[-2, 2:4].copy_(shared_tail.expand(2))
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