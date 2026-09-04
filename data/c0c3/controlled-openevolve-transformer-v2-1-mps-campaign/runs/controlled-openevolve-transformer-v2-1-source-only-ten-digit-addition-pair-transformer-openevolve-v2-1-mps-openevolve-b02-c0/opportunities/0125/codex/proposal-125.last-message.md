MECHANISM: Second-position residual-stream common-mode gauge

HYPOTHESIS: Representing positional row 1 in the same seven-dimensional zero-mean basis as row 0 will reduce the model to 1,573 parameters while retaining at least 99% accuracy, because its removed all-ones component is eliminated by every downstream LayerNorm path.

INTENDED_EDIT: Compact positional row 1 from eight weights to seven orthonormal coordinates, reconstruct it during forward passes, and preserve full-matrix initialization draws for RNG alignment.

EVIDENCE: Gauging positional row 0 reduced the verified design to 1,574 parameters with 99.98% accuracy, while eight other positional rows already use the same basis; this makes another untouched positional row the cleanest extension of the most recently successful mechanism.

<<<<<<< SEARCH
        self.initial_gauge_index = 0
        self.initial_gauge_end_index = embedding_dim - 1
        self.earlier_gauge_index = (
            self.initial_gauge_end_index
            + self.earlier_gauge_flat_start
            - embedding_dim
        )
=======
        self.initial_gauge_index = 0
        self.initial_gauge_end_index = embedding_dim - 1
        self.next_gauge_index = self.initial_gauge_end_index
        self.next_gauge_end_index = self.next_gauge_index + embedding_dim - 1
        self.earlier_gauge_index = (
            self.next_gauge_end_index
            + self.earlier_gauge_flat_start
            - 2 * embedding_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        initial_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[:embedding_dim]
        )
        earlier_gauge_coords = (
=======
        initial_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[:embedding_dim]
        )
        next_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[embedding_dim : 2 * embedding_dim]
        )
        earlier_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                initial_gauge_coords,
                flat_weight[embedding_dim : self.earlier_gauge_flat_start],
                earlier_gauge_coords,
=======
                initial_gauge_coords,
                next_gauge_coords,
                flat_weight[2 * embedding_dim : self.earlier_gauge_flat_start],
                earlier_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
        initial_gauge_row = (
            self.gauge_basis
            @ self.weight[
                self.initial_gauge_index : self.initial_gauge_end_index
            ]
        )
        earlier_gauge_row = (
=======
        initial_gauge_row = (
            self.gauge_basis
            @ self.weight[
                self.initial_gauge_index : self.initial_gauge_end_index
            ]
        )
        next_gauge_row = (
            self.gauge_basis
            @ self.weight[self.next_gauge_index : self.next_gauge_end_index]
        )
        earlier_gauge_row = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                initial_gauge_row,
                self.weight[
                    self.initial_gauge_end_index : self.earlier_gauge_index
                ],
                earlier_gauge_row,
=======
                initial_gauge_row,
                next_gauge_row,
                self.weight[
                    self.next_gauge_end_index : self.earlier_gauge_index
                ],
                earlier_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            initial_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[: module.embedding_dim]
            )
            earlier_gauge_coords = (
=======
            initial_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[: module.embedding_dim]
            )
            next_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.embedding_dim : 2 * module.embedding_dim
                ]
            )
            earlier_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                    initial_gauge_coords,
                    flat_weight[
                        module.embedding_dim : module.earlier_gauge_flat_start
                    ],
                    earlier_gauge_coords,
=======
                    initial_gauge_coords,
                    next_gauge_coords,
                    flat_weight[
                        2 * module.embedding_dim : module.earlier_gauge_flat_start
                    ],
                    earlier_gauge_coords,
>>>>>>> REPLACE