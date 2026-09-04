MECHANISM: First-head penultimate zero-bias query/key scale chart

HYPOTHESIS: Normalizing the first head’s penultimate zero-bias query row will reduce the verified 1494-parameter model to 1493 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add a max-pivot chart for the first head’s penultimate query row, omit one learned coordinate, and absorb its initialization scale into the matching key row.

EVIDENCE: The corresponding second-head penultimate chart achieved 99.88% accuracy at 1494 parameters; testing the same row position in the other head is the closest supported reduction, while differing from the failed first-head final-row chart.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head. Both zero-bias trailing coordinates of
        # the second head additionally use diagonal query/key scale gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head. Both penultimate zero-bias query rows and
        # the second head's final row use diagonal query/key scale gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_first_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
=======
        q_first_penultimate_pivot = int(
            self.q_first_penultimate_pivot.item()
        )
        q_first_penultimate_chart = torch.cat(
            (
                self.q_first_penultimate_weight[
                    :q_first_penultimate_pivot
                ],
                self.q_first_penultimate_weight.new_full(
                    (1,), 1.0
                ),
                self.q_first_penultimate_weight[
                    q_first_penultimate_pivot:
                ],
            )
        )
        q_first_penultimate_relative = (
            q_first_penultimate_chart
            * (
                (0.02 * math.sqrt(d_model - 1))
                / q_first_penultimate_chart.norm()
            )
        )

        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_penultimate_row = d_model - 2
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_penultimate_row],
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_penultimate_row:],
            ),
            dim=0,
        )
=======
        q_first_penultimate_row = self.head_dim - 2
        q_penultimate_row = d_model - 2
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_first_penultimate_row],
                q_first_penultimate_relative.unsqueeze(0),
                self.qkv.weight[
                    q_first_penultimate_row:q_penultimate_row - 1
                ],
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_penultimate_row - 1:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_penultimate = cfg.d_model - 2
            q_penultimate_free = relative_qkv_weight[q_penultimate]
=======
            q_first_penultimate = block.attn.head_dim - 2
            q_first_penultimate_free = relative_qkv_weight[
                q_first_penultimate
            ]
            q_first_penultimate_pivot = int(
                q_first_penultimate_free.abs().argmax().item()
            )
            q_first_penultimate_pivot_value = (
                q_first_penultimate_free[
                    q_first_penultimate_pivot
                ]
            )
            q_first_penultimate_chart = (
                q_first_penultimate_free
                / q_first_penultimate_pivot_value
            )
            q_first_penultimate_gauge_norm = (
                0.02
                * math.sqrt(q_first_penultimate_free.numel())
            )
            q_first_penultimate_scale = (
                q_first_penultimate_pivot_value.sign()
                * q_first_penultimate_free.norm()
                / q_first_penultimate_gauge_norm
            )
            key_first_penultimate = (
                cfg.d_model + q_first_penultimate
            )
            full_qkv_weight[key_first_penultimate] = (
                q_first_penultimate_scale
                * full_qkv_weight[key_first_penultimate]
            )
            block.attn.q_first_penultimate_pivot.fill_(
                q_first_penultimate_pivot
            )

            q_penultimate = cfg.d_model - 2
            q_penultimate_free = relative_qkv_weight[q_penultimate]
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_penultimate],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_penultimate_weight = nn.Parameter(
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[
                            :q_first_penultimate
                        ],
                        relative_qkv_weight[
                            q_first_penultimate + 1:q_penultimate
                        ],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_first_penultimate_weight = nn.Parameter(
                torch.cat(
                    (
                        q_first_penultimate_chart[
                            :q_first_penultimate_pivot
                        ],
                        q_first_penultimate_chart[
                            q_first_penultimate_pivot + 1:
                        ],
                    )
                )
            )
            block.attn.q_penultimate_weight = nn.Parameter(
>>>>>>> REPLACE