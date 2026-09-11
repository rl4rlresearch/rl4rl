MECHANISM: Partial companion-key coordinate alignment

HYPOTHESIS: Fixing the leading tail coefficient of head 1’s first key channel will reduce the model from 991 to 990 parameters while retaining at least 99% accuracy, because it incrementally extends the successful coordinate-aligned key constraint without reducing the two-dimensional score rank.

INTENDED_EDIT: Omit one coefficient from head 1’s first key-tail row, reconstruct it as zero, and project the virtual optimizer state onto both key constraints after every update.

EVIDENCE: Fixing all five tail coefficients of head 1’s second key channel achieved 99.94% at 991 parameters; removing only one neighboring coefficient is the smallest direct test of whether that demonstrated coordinate alignment extends to the companion channel.

<<<<<<< SEARCH
        # The final flattened row is head 1's second key channel. Its
        # identity prefix remains active while its tail is fixed at zero.
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head * self.score_dim - 1,
                reduced_width - self.score_dim,
            )
        )
=======
        # Head 1's second key channel has a fixed-zero tail. The leading
        # tail coefficient of its companion channel is also fixed at zero.
        self.key_tail = nn.Parameter(
            torch.empty(
                (n_head * self.score_dim - 1)
                * (reduced_width - self.score_dim)
                - 1
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_key_tail = canonical_key_tail.reshape(
                -1, self.d_model - 1 - self.score_dim
            )
            flat_key_tail[-1].zero_()
            self.key_tail.copy_(flat_key_tail[:-1])
            canonical_key_tail = flat_key_tail.view(
=======
            flat_key_tail = canonical_key_tail.reshape(
                -1, self.d_model - 1 - self.score_dim
            )
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
            canonical_key_tail = flat_key_tail.view(
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_tail = torch.cat(
            [
                self.key_tail,
                self.key_tail.new_zeros(
                    1, self.key_tail.shape[-1]
                ),
            ],
            dim=0,
        ).view(
            self.n_head,
            self.score_dim,
            reduced_x.shape[-1] - self.score_dim,
        )
=======
        tail_width = reduced_x.shape[-1] - self.score_dim
        learned_boundary = (
            (self.n_head * self.score_dim - 2) * tail_width
        )
        key_tail = torch.cat(
            [
                self.key_tail[:learned_boundary],
                self.key_tail.new_zeros(1),
                self.key_tail[learned_boundary:],
                self.key_tail.new_zeros(tail_width),
            ],
            dim=0,
        ).view(
            self.n_head,
            self.score_dim,
            tail_width,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        canonical_key_tail_grad = torch.cat(
            [
                stored_key_tail_grad,
                stored_key_tail_grad.new_zeros(1, tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        canonical_key_tail = torch.cat(
            [
                key_tail_param.detach(),
                key_tail_param.new_zeros(1, tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
=======
        learned_boundary = (
            (n_head * head_dim - 2) * tail_width
        )
        canonical_key_tail_grad = torch.cat(
            [
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
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_key_tail = canonical_key_tail.reshape(
                -1, reduced_width - head_dim
            )
            flat_key_tail[-1].zero_()
            key_tail_param.copy_(flat_key_tail[:-1])
            canonical_key_tail = flat_key_tail.view(
=======
            flat_key_tail = canonical_key_tail.reshape(
                -1, reduced_width - head_dim
            )
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
            canonical_key_tail = flat_key_tail.view(
>>>>>>> REPLACE