MECHANISM: Axis-aligned third-position gauge with trajectory-preserving runtime reductions

HYPOTHESIS: Anchoring positional row 2’s final coordinate and reducing validation and sampling overhead will produce a 1,572-parameter model with at least 99% accuracy that completes 4,999 training steps within the verification limit.

INTENDED_EDIT: Store positional row 2 as seven differences from its omitted final coordinate, reconstruct that coordinate as zero, preserve full-matrix initialization draws, bulk-convert sampled operands without changing samples, and evaluate every 1,000 steps.

EVIDENCE: Positional rows 0 and 1 reached 99.98% and 99.99% accuracy at 1,574 and 1,573 parameters; all row-2 reductions timed out rather than failing accuracy, so retrying the lowest-overhead axis-aligned gauge with training-trajectory-neutral runtime savings directly tests the remaining uncertainty.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
=======
        flat_weight = self.weight.detach().flatten()
        self.middle_anchor_flat_index = 3 * embedding_dim - 1
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.next_gauge_index = self.initial_gauge_end_index
        self.next_gauge_end_index = self.next_gauge_index + embedding_dim - 1
        self.earlier_gauge_index = (
            self.next_gauge_end_index
            + self.earlier_gauge_flat_start
            - 2 * embedding_dim
        )
=======
        self.next_gauge_index = self.initial_gauge_end_index
        self.next_gauge_end_index = self.next_gauge_index + embedding_dim - 1
        self.middle_anchor_index = (
            self.next_gauge_end_index + embedding_dim - 1
        )
        self.earlier_gauge_index = (
            self.next_gauge_end_index
            + self.earlier_gauge_flat_start
            - 2 * embedding_dim
            - 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        next_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[embedding_dim : 2 * embedding_dim]
        )
        earlier_gauge_coords = (
=======
        next_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[embedding_dim : 2 * embedding_dim]
        )
        middle_anchor_coords = (
            flat_weight[
                2 * embedding_dim : self.middle_anchor_flat_index
            ]
            - flat_weight[self.middle_anchor_flat_index]
        )
        earlier_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                initial_gauge_coords,
                next_gauge_coords,
                flat_weight[2 * embedding_dim : self.earlier_gauge_flat_start],
                earlier_gauge_coords,
=======
                initial_gauge_coords,
                next_gauge_coords,
                middle_anchor_coords,
                flat_weight[
                    self.middle_anchor_flat_index + 1 :
                    self.earlier_gauge_flat_start
                ],
                earlier_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
                initial_gauge_row,
                next_gauge_row,
                self.weight[
                    self.next_gauge_end_index : self.earlier_gauge_index
                ],
                earlier_gauge_row,
=======
                initial_gauge_row,
                next_gauge_row,
                self.weight[
                    self.next_gauge_end_index : self.middle_anchor_index
                ],
                self.weight.new_zeros(1),
                self.weight[
                    self.middle_anchor_index : self.earlier_gauge_index
                ],
                earlier_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            next_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.embedding_dim : 2 * module.embedding_dim
                ]
            )
            earlier_gauge_coords = (
=======
            next_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.embedding_dim : 2 * module.embedding_dim
                ]
            )
            middle_anchor_coords = (
                flat_weight[
                    2 * module.embedding_dim :
                    module.middle_anchor_flat_index
                ]
                - flat_weight[module.middle_anchor_flat_index]
            )
            earlier_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                    initial_gauge_coords,
                    next_gauge_coords,
                    flat_weight[
                        2 * module.embedding_dim : module.earlier_gauge_flat_start
                    ],
                    earlier_gauge_coords,
=======
                    initial_gauge_coords,
                    next_gauge_coords,
                    middle_anchor_coords,
                    flat_weight[
                        module.middle_anchor_flat_index + 1 :
                        module.earlier_gauge_flat_start
                    ],
                    earlier_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
=======
        for i, (ai, bi) in enumerate(zip(a.tolist(), b.tolist())):
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE