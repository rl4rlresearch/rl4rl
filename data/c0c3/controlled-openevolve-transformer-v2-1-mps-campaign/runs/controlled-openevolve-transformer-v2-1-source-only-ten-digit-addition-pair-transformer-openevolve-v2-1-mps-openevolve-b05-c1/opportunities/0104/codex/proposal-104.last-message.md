MECHANISM: Orthogonal value/output gauge reuse for final normalization scale

HYPOTHESIS: Encoding the remaining final-LayerNorm scale as one plus a value/output rotation-gauge coefficient will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy, without adding a costly reduction to the training hot path.

INTENDED_EDIT: Orthogonally rotate the final two value channels during initialization so one coefficient is zero, inverse-rotate the corresponding output-projection columns to preserve the initialized function, remove the dedicated final-LayerNorm scale, and derive that scale from the zero-centered gauge coefficient.

EVIDENCE: Orthonormally conditioned null-direction reuse retained 100% accuracy at both 1,267 and 1,266 parameters, whereas reductions adding another column-mean operation or coefficient reconstruction repeatedly timed out. This uses the same unit-gradient, initialization-preserving principle through an independent exact value/output basis symmetry.

<<<<<<< SEARCH
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
=======
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
        self.register_buffer(
            "value_rotation",
            torch.empty(2),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
=======
            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])

            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
            first_value = self.v_weight[-2].clone()
            second_value = self.v_weight[-1].clone()
            radius = torch.sqrt(
                first_value[-1].square() + second_value[-1].square()
            )
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first_value[-1] / radius
            sine = second_value[-1] / radius
            self.v_weight[-2].copy_(
                cosine * first_value + sine * second_value
            )
            self.v_weight[-1].copy_(
                -sine * first_value + cosine * second_value
            )
            self.value_rotation.copy_(torch.stack((cosine, sine)))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
=======
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                shared_scales[2:3],
                shared_scales.new_ones(1),
                shared_scales[1:2],
                shared_scales.new_ones(2),
                shared_scales[:1],
                shared_scales.new_ones(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, SingleRotationGaugeQKV):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        if isinstance(module, SingleRotationGaugeQKV):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, CausalSelfAttention):
            cosine, sine = module.qkv.value_rotation.unbind()
            first_column = module.proj.weight[:, -2].clone()
            second_column = module.proj.weight[:, -1].clone()
            module.proj.weight[:, -2].copy_(
                cosine * first_column + sine * second_column
            )
            module.proj.weight[:, -1].copy_(
                -sine * first_column + cosine * second_column
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
=======
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
                1.0 + self.blocks[-1].attn.qkv.v_weight[-1, -1],
            )
        )
>>>>>>> REPLACE