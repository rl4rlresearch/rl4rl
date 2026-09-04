MECHANISM: First-head bias-preserving query/key shear

HYPOTHESIS: Applying one exact shear from the first head’s neighboring bias-bearing query row into its freely learned zero-bias target row will reduce the model to 1489 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Omit one pivot coordinate from the first head’s second query row, reconstruct it as zero, and absorb the inverse initialization shear into the matching key row.

EVIDENCE: The analogous second-head bias-preserving shears achieved 99.91% at 1490 parameters. This tests the same successful gauge in the untouched first head without imposing the failed first-head target-row normalization or the failed third value/output shear.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge, while the neighboring bias-bearing row is
        # sheared against both of them without changing its learned bias.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Each head's neighboring bias-bearing query row is
        # sheared against a zero-bias target; the second head additionally
        # uses the full scale/shear gauge of its two zero-bias rows.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_shear_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_first_shear_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )

        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
=======
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )

        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_first_shear_relative = torch.cat(
            (
                self.q_first_shear_weight[:q_first_shear_pivot],
                self.q_first_shear_weight.new_zeros(1),
                self.q_first_shear_weight[q_first_shear_pivot:],
            )
        )

        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_shear_row = d_model - 3
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_shear_row],
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_shear_row:],
            ),
            dim=0,
        )
=======
        q_first_shear_row = self.head_dim - 3
        q_shear_row = d_model - 3
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_first_shear_row],
                q_first_shear_relative.unsqueeze(0),
                self.qkv.weight[
                    q_first_shear_row:q_shear_row - 1
                ],
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_shear_row - 1:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )

            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
=======
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )

            q_first_shear = block.attn.head_dim - 3
            q_first_target = block.attn.head_dim - 1
            q_first_shear_free = relative_qkv_weight[
                q_first_shear
            ]
            q_first_target_free = relative_qkv_weight[
                q_first_target
            ]
            q_first_shear_pivot = int(
                q_first_target_free.abs().argmax().item()
            )
            q_first_shear_coefficient = (
                q_first_shear_free[q_first_shear_pivot]
                / q_first_target_free[q_first_shear_pivot]
            )
            q_first_shear_chart = (
                q_first_shear_free
                - q_first_shear_coefficient
                * q_first_target_free
            )
            q_first_shear_chart[q_first_shear_pivot] = 0.0
            key_first_shear = cfg.d_model + q_first_shear
            key_first_target = cfg.d_model + q_first_target
            full_qkv_weight[key_first_target] = (
                full_qkv_weight[key_first_target]
                + q_first_shear_coefficient
                * full_qkv_weight[key_first_shear]
            )
            block.attn.q_first_shear_pivot.fill_(
                q_first_shear_pivot
            )

            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_shear],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            query_free = [
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_first_shear],
                        relative_qkv_weight[
                            q_first_shear + 1:q_shear
                        ],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_first_shear_weight = nn.Parameter(
                torch.cat(
                    (
                        q_first_shear_chart[:q_first_shear_pivot],
                        q_first_shear_chart[
                            q_first_shear_pivot + 1:
                        ],
                    )
                )
            )
            query_free = [
>>>>>>> REPLACE