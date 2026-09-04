MECHANISM: First-head mutual bias-bearing query/key shear

HYPOTHESIS: Shearing one freely biased first-head query row against the other will reduce the verified 1487-parameter model to 1486 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Omit one pivot coordinate from the first head’s leading biased query row and apply the inverse initialization shear to the other biased row’s key weights.

EVIDENCE: The analogous mutual shear between freely biased second-head rows achieved 99.99% at 1489 parameters; unlike the failed first-head bias-to-zero-bias shear, this preserves independently learned biases on both transformed rows.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's terminal zero-bias row uses a scale
        # gauge and the complementary shear against its zero-bias neighbor.
        # The second head uses its qualified scale and shear construction.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head uses a mutual shear between its freely
        # biased rows, plus the qualified terminal-row scale and complementary
        # zero-bias shear. The second head retains its qualified construction.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_lead_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "q_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        self.register_buffer(
            "q_lead_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())

        q_first_free = [
=======
        q_lead_pivot = int(self.q_lead_pivot.item())
        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())

        q_lead_relative = torch.cat(
            (
                self.q_lead_weight[:q_lead_pivot],
                self.q_lead_weight.new_zeros(1),
                self.q_lead_weight[q_lead_pivot:],
            )
        )
        q_first_free = [
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_first_target = self.head_dim - 1
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_first_target],
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_first_target:],
            ),
            dim=0,
        )
=======
        q_first_target = self.head_dim - 1
        qkv_rows = torch.cat(
            (
                q_lead_relative.unsqueeze(0),
                self.qkv.weight[:q_first_target - 1],
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_first_target - 1:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.q_first_shear_pivot.fill_(
                q_first_shear_pivot
            )

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
=======
            block.attn.q_first_shear_pivot.fill_(
                q_first_shear_pivot
            )

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_lead = 0
            q_basis = 1
            q_lead_free = relative_qkv_weight[q_lead]
            q_basis_free = relative_qkv_weight[q_basis]
            q_lead_pivot = int(
                q_basis_free.abs().argmax().item()
            )
            q_lead_shear = (
                q_lead_free[q_lead_pivot]
                / q_basis_free[q_lead_pivot]
            )
            q_lead_chart = (
                q_lead_free - q_lead_shear * q_basis_free
            )
            q_lead_chart[q_lead_pivot] = 0.0

            key_lead = cfg.d_model + q_lead
            key_basis = cfg.d_model + q_basis
            key_lead_free = full_qkv_weight[key_lead].clone()
            full_qkv_weight[key_basis] = (
                full_qkv_weight[key_basis]
                + q_lead_shear * key_lead_free
            )
            block.attn.q_lead_pivot.fill_(q_lead_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
>>>>>>> REPLACE

<<<<<<< SEARCH
                        relative_qkv_weight[:q_first_target],
                        relative_qkv_weight[q_target + 1:],
=======
                        relative_qkv_weight[
                            q_lead + 1:q_first_target
                        ],
                        relative_qkv_weight[q_target + 1:],
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.q_first_weight = nn.Parameter(
                q_first_chart[q_first_free_coordinates].clone()
            )
=======
            block.attn.q_lead_weight = nn.Parameter(
                torch.cat(
                    (
                        q_lead_chart[:q_lead_pivot],
                        q_lead_chart[q_lead_pivot + 1:],
                    )
                )
            )
            block.attn.q_first_weight = nn.Parameter(
                q_first_chart[q_first_free_coordinates].clone()
            )
>>>>>>> REPLACE