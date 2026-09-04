MECHANISM: Extended companion-key coordinate alignment

HYPOTHESIS: Fixing the third tail coefficient of head 1’s first key channel will reduce the model from 989 to 988 parameters while retaining at least 99% accuracy, because fixing its first two adjacent coefficients achieved 99.98% without reducing the proven two-dimensional attention score rank.

INTENDED_EDIT: Omit the first three tail coefficients from head 1’s first key channel, reconstruct them as zeros, and project the virtual optimizer state onto the extended constraint after every update.

EVIDENCE: Fixing the first companion-channel coefficient achieved 99.97% at 990 parameters, and extending the same contiguous constraint to two coefficients improved to 99.98% at 989 parameters; removing the adjacent third coefficient is the smallest informative continuation.

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
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(2),
                stored_key_tail_grad[learned_boundary:],
                stored_key_tail_grad.new_zeros(tail_width),
=======
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(3),
                stored_key_tail_grad[learned_boundary:],
                stored_key_tail_grad.new_zeros(tail_width),
>>>>>>> REPLACE

<<<<<<< SEARCH
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(2),
                detached_key_tail[learned_boundary:],
                detached_key_tail.new_zeros(tail_width),
=======
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(3),
                detached_key_tail[learned_boundary:],
                detached_key_tail.new_zeros(tail_width),
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