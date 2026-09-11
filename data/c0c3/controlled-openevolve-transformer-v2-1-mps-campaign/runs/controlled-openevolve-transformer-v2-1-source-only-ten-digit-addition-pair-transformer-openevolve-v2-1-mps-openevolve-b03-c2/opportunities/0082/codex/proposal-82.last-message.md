MECHANISM: First-head terminal query/key scale gauge atop the qualified mutual-shear chart

HYPOTHESIS: Reproducing the verified 1489-parameter mutual-shear design and normalizing the untouched final zero-bias query row of the first head will yield 1488 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add Reference Design 1’s mutual bias-bearing second-head shear, then omit one coordinate and the scalar scale from the first head’s final query row while transferring its initialization scale to the matching key row.

EVIDENCE: Reference Design 1 achieved 99.99% accuracy at 1489 parameters. Although a first-head penultimate-row chart failed, the terminal zero-bias row remains untested, making this a row-specific test of the same exact diagonal query/key gauge already successful on second-head zero-bias rows.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge, while the neighboring bias-bearing row is
        # sheared against both of them without changing its learned bias.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
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
        # output bias. The first head's final zero-bias query row uses a
        # diagonal scale gauge. The second head's zero-bias rows use their
        # scale/shear gauge, one biased row is sheared against both, and the
        # other biased row is sheared against that freely biased anchor.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "q_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_pivot",
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
        q_target_pivot = int(self.q_target_pivot.item())
=======
        q_first_pivot = int(self.q_first_pivot.item())
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_shear_relative = self.q_shear_weight.new_zeros(
            d_model - 1
        )
        q_shear_relative[query_free] = self.q_shear_weight
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
        q_first_chart = torch.cat(
            (
                self.q_first_weight[:q_first_pivot],
                self.q_first_weight.new_full((1,), 1.0),
                self.q_first_weight[q_first_pivot:],
            )
        )
        q_first_relative = q_first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_first_chart.norm()
        )
        q_anchor_relative = torch.cat(
            (
                self.q_anchor_weight[:q_anchor_pivot],
                self.q_anchor_weight.new_zeros(1),
                self.q_anchor_weight[q_anchor_pivot:],
            )
        )
        q_shear_relative = self.q_shear_weight.new_zeros(
            d_model - 1
        )
        q_shear_relative[query_free] = self.q_shear_weight
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj_last_pivot.fill_(last_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )

            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_shear_free = relative_qkv_weight[q_shear]
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_target_free = relative_qkv_weight[q_target]
=======
            block.attn.proj_last_pivot.fill_(last_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_first_target = block.attn.head_dim - 1
            q_first_free = relative_qkv_weight[q_first_target]
            q_first_pivot = int(
                q_first_free.abs().argmax().item()
            )
            q_first_pivot_value = q_first_free[q_first_pivot]
            q_first_chart = q_first_free / q_first_pivot_value
            q_first_gauge_norm = (
                0.02 * math.sqrt(q_first_free.numel())
            )
            q_first_scale = (
                q_first_pivot_value.sign()
                * q_first_free.norm()
                / q_first_gauge_norm
            )
            key_first_target = cfg.d_model + q_first_target
            full_qkv_weight[key_first_target] = (
                q_first_scale * full_qkv_weight[key_first_target]
            )
            block.attn.q_first_pivot.fill_(q_first_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_anchor_free = relative_qkv_weight[q_anchor]
            q_shear_free = relative_qkv_weight[q_shear]
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_target_free = relative_qkv_weight[q_target]
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_shear_chart[q_penultimate_pivot] = 0.0
            q_shear_chart[q_target_pivot] = 0.0

            key_shear = cfg.d_model + q_shear
            key_penultimate = cfg.d_model + q_penultimate
            key_target = cfg.d_model + q_target
            key_penultimate_free = full_qkv_weight[
                key_penultimate
            ].clone()
            key_target_free = full_qkv_weight[key_target].clone()
            key_target_sheared = (
                key_target_free
                + first_shear * key_penultimate_free
            )
            key_penultimate_sheared = (
                key_penultimate_free
                + second_shear * key_target_sheared
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear
                * full_qkv_weight[key_shear]
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear
                * full_qkv_weight[key_shear]
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)
=======
            q_shear_chart[q_penultimate_pivot] = 0.0
            q_shear_chart[q_target_pivot] = 0.0

            q_anchor_pivot = int(
                q_shear_chart.abs().argmax().item()
            )
            anchor_shear = (
                q_anchor_free[q_anchor_pivot]
                / q_shear_chart[q_anchor_pivot]
            )
            q_anchor_chart = (
                q_anchor_free - anchor_shear * q_shear_chart
            )
            q_anchor_chart[q_anchor_pivot] = 0.0

            key_anchor = cfg.d_model + q_anchor
            key_shear = cfg.d_model + q_shear
            key_penultimate = cfg.d_model + q_penultimate
            key_target = cfg.d_model + q_target
            key_anchor_free = full_qkv_weight[key_anchor].clone()
            key_shear_free = full_qkv_weight[key_shear].clone()
            key_penultimate_free = full_qkv_weight[
                key_penultimate
            ].clone()
            key_target_free = full_qkv_weight[key_target].clone()
            key_target_sheared = (
                key_target_free
                + first_shear * key_penultimate_free
            )
            key_penultimate_sheared = (
                key_penultimate_free
                + second_shear * key_target_sheared
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear * key_shear_free
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
            )
            full_qkv_weight[key_shear] = (
                key_shear_free + anchor_shear * key_anchor_free
            )
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)
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
                        relative_qkv_weight[:q_first_target],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_first_weight = nn.Parameter(
                torch.cat(
                    (
                        q_first_chart[:q_first_pivot],
                        q_first_chart[q_first_pivot + 1:],
                    )
                )
            )
            block.attn.q_anchor_weight = nn.Parameter(
                torch.cat(
                    (
                        q_anchor_chart[:q_anchor_pivot],
                        q_anchor_chart[q_anchor_pivot + 1:],
                    )
                )
            )
            query_free = [
>>>>>>> REPLACE