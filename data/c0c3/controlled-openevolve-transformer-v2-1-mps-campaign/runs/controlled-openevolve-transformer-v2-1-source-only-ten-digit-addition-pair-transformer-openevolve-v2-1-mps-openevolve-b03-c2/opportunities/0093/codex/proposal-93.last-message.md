MECHANISM: Max-pivot orthogonal value/output neighbor gauge

HYPOTHESIS: Constraining the first head’s neighboring projection row to be orthogonal to its normalized target row will reduce the model to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one learned coordinate from that projection row, reconstruct it from the orthogonality constraint, absorb the initialization shear into the matching value row, and update optimizer-side projection reconstruction.

EVIDENCE: The current normalized orthogonal-query design reached 99.92% at 1485 parameters, and max-pivot orthogonality previously improved an analogous query reduction from 97.76% to 99.79%; the prior value/output attempt was not tested because its search matched multiple locations.

<<<<<<< SEARCH
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.register_buffer(
            "proj_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_last_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.register_buffer(
            "proj_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_neighbor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_last_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_relative = torch.cat(
            (head_free, head_free.new_zeros(2))
        )
        last_relative = torch.cat(
            (last_free, last_free.new_zeros(2))
        )
        split = self.head_dim - 1
        weight_rows = torch.cat(
            (
                self.proj.weight[:split],
                head_relative.unsqueeze(0),
                self.proj.weight[split:],
                last_relative.unsqueeze(0),
            ),
            dim=0,
        )
=======
        head_relative = torch.cat(
            (head_free, head_free.new_zeros(2))
        )
        neighbor_pivot = int(
            self.proj_neighbor_pivot.item()
        )
        neighbor_coordinates = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != neighbor_pivot
        ]
        neighbor_relative = (
            self.proj_neighbor_weight.new_zeros(d_model - 1)
        )
        neighbor_relative[neighbor_coordinates] = (
            self.proj_neighbor_weight
        )
        neighbor_relative[neighbor_pivot] = -(
            neighbor_relative[neighbor_coordinates]
            * head_relative[neighbor_coordinates]
        ).sum() / head_relative[neighbor_pivot]
        last_relative = torch.cat(
            (last_free, last_free.new_zeros(2))
        )
        split = self.head_dim - 2
        weight_rows = torch.cat(
            (
                self.proj.weight[:split],
                neighbor_relative.unsqueeze(0),
                head_relative.unsqueeze(0),
                self.proj.weight[split:],
                last_relative.unsqueeze(0),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_qkv_weight[value_start + first_target] = (
                head_scale
                * full_qkv_weight[value_start + first_target]
            )
            block.attn.proj_head_pivot.fill_(head_pivot)

            last_target = cfg.d_model - 1
=======
            full_qkv_weight[value_start + first_target] = (
                head_scale
                * full_qkv_weight[value_start + first_target]
            )
            block.attn.proj_head_pivot.fill_(head_pivot)

            head_relative = torch.cat(
                (
                    head_chart * (
                        head_gauge_norm / head_chart.norm()
                    ),
                    head_chart.new_zeros(2),
                )
            )
            neighbor_row = first_target - 1
            neighbor_free = relative_proj_weight[neighbor_row]
            neighbor_shear = (
                neighbor_free * head_relative
            ).sum() / head_relative.square().sum()
            neighbor_orthogonal = (
                neighbor_free - neighbor_shear * head_relative
            )
            neighbor_pivot = int(
                head_relative.abs().argmax().item()
            )
            full_qkv_weight[value_start + first_target] = (
                full_qkv_weight[value_start + first_target]
                + neighbor_shear
                * full_qkv_weight[value_start + neighbor_row]
            )
            block.attn.proj_neighbor_pivot.fill_(
                neighbor_pivot
            )

            last_target = cfg.d_model - 1
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[:first_target],
                        relative_proj_weight[
                            first_target + 1:-1
                        ],
                    ),
                    dim=0,
                )
            )
            block.attn.proj_head_weight = nn.Parameter(
=======
            block.attn.proj.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[:neighbor_row],
                        relative_proj_weight[
                            first_target + 1:-1
                        ],
                    ),
                    dim=0,
                )
            )
            neighbor_coordinates = [
                coordinate
                for coordinate in range(
                    neighbor_orthogonal.numel()
                )
                if coordinate != neighbor_pivot
            ]
            block.attn.proj_neighbor_weight = nn.Parameter(
                neighbor_orthogonal[
                    neighbor_coordinates
                ].clone()
            )
            block.attn.proj_head_weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
def reconstruct_attention_output_weight(
    weight,
    head_weight,
    last_weight,
    head_dim,
    head_pivot,
    last_pivot,
):
=======
def reconstruct_attention_output_weight(
    weight,
    head_weight,
    neighbor_weight,
    last_weight,
    head_dim,
    head_pivot,
    neighbor_pivot,
    last_pivot,
):
>>>>>>> REPLACE

<<<<<<< SEARCH
    head_free = head_chart * (
        (0.02 * math.sqrt(head_weight.numel() + 1))
        / head_chart.norm()
    )
    last_chart = torch.cat(
        (
            last_weight[:last_pivot],
            last_weight.new_full((1,), 1.0),
            last_weight[last_pivot:],
        )
    )
    last_free = last_chart * (
        (0.02 * math.sqrt(last_weight.numel() + 1))
        / last_chart.norm()
    )
    head_relative = torch.cat(
        (head_free, head_free.new_zeros(2))
    )
    last_relative = torch.cat(
        (last_free, last_free.new_zeros(2))
    )
    split = head_dim - 1
    rows = torch.cat(
        (
            weight[:split],
            head_relative.unsqueeze(0),
            weight[split:],
            last_relative.unsqueeze(0),
        ),
        dim=0,
    )
=======
    head_free = head_chart * (
        (0.02 * math.sqrt(head_weight.numel() + 1))
        / head_chart.norm()
    )
    head_relative = torch.cat(
        (head_free, head_free.new_zeros(2))
    )
    neighbor_coordinates = [
        coordinate
        for coordinate in range(head_relative.numel())
        if coordinate != neighbor_pivot
    ]
    neighbor_relative = neighbor_weight.new_zeros(
        head_relative.numel()
    )
    neighbor_relative[neighbor_coordinates] = neighbor_weight
    neighbor_relative[neighbor_pivot] = -(
        neighbor_relative[neighbor_coordinates]
        * head_relative[neighbor_coordinates]
    ).sum() / head_relative[neighbor_pivot]
    last_chart = torch.cat(
        (
            last_weight[:last_pivot],
            last_weight.new_full((1,), 1.0),
            last_weight[last_pivot:],
        )
    )
    last_free = last_chart * (
        (0.02 * math.sqrt(last_weight.numel() + 1))
        / last_chart.norm()
    )
    last_relative = torch.cat(
        (last_free, last_free.new_zeros(2))
    )
    split = head_dim - 2
    rows = torch.cat(
        (
            weight[:split],
            neighbor_relative.unsqueeze(0),
            head_relative.unsqueeze(0),
            weight[split:],
            last_relative.unsqueeze(0),
        ),
        dim=0,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias_grads = []
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            last_pivot,
        ) in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                full_proj_grad = torch.cat(
                    (
                        proj_bias.grad.detach(),
                        -proj_bias.grad.detach().sum().view(1),
                    )
                )
                full_proj_weight = (
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_head_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                        head_pivot,
                        last_pivot,
                    )
                )
                grad = (
=======
        value_bias_grads = []
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_neighbor_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            neighbor_pivot,
            last_pivot,
        ) in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                full_proj_grad = torch.cat(
                    (
                        proj_bias.grad.detach(),
                        -proj_bias.grad.detach().sum().view(1),
                    )
                )
                full_proj_weight = (
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_head_weight.detach(),
                        proj_neighbor_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                        head_pivot,
                        neighbor_pivot,
                        last_pivot,
                    )
                )
                grad = (
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            last_pivot,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
=======
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_neighbor_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            neighbor_pivot,
            last_pivot,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_delta = (
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                    head_pivot,
                    last_pivot,
                )
                @ omitted_value
            )
=======
            full_delta = (
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_neighbor_weight,
                    proj_last_weight,
                    head_dim,
                    head_pivot,
                    neighbor_pivot,
                    last_pivot,
                )
                @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
        head_pivot,
        last_pivot,
    ) in value_bias_specs:
=======
    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_neighbor_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
        head_pivot,
        neighbor_pivot,
        last_pivot,
    ) in value_bias_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
            head_pivot,
            last_pivot,
        )
        omitted_grad = (
=======
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_neighbor_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
            head_pivot,
            neighbor_pivot,
            last_pivot,
        )
        omitted_grad = (
>>>>>>> REPLACE

<<<<<<< SEARCH
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
            int(block.attn.proj_head_pivot.item()),
            int(block.attn.proj_last_pivot.item()),
        )
        for block in model.blocks
    ]
=======
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_neighbor_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
            int(block.attn.proj_head_pivot.item()),
            int(block.attn.proj_neighbor_pivot.item()),
            int(block.attn.proj_last_pivot.item()),
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE