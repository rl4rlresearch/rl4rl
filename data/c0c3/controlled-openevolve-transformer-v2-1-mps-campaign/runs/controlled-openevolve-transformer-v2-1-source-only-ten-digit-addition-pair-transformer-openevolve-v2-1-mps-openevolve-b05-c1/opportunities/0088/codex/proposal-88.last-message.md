MECHANISM: Single value-output rotational gauge

HYPOTHESIS: Fixing one rotationally redundant value-projection coefficient will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy, because an orthogonal rotation of two value coordinates within a head can be exactly canceled in the attention output projection.

INTENDED_EDIT: Rotate the first two value rows at initialization, omit the second row’s first coefficient, reconstruct it as zero during forward passes, and counter-rotate the corresponding output-projection columns to preserve the initialized model function exactly.

EVIDENCE: A single function-preserving query/key rotation retained 100% accuracy at 1,271 parameters, but adding further constraints within the increasingly restricted query/key charts failed; this tests one parameter from a separate exact attention symmetry while retaining function-preserving initialization.

<<<<<<< SEARCH
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
=======
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_first = nn.Parameter(torch.empty(1, in_features))
        self.v_second_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.v_rest = nn.Parameter(torch.empty(d_model - 2, in_features))
        self.register_buffer(
            "value_rotation",
            torch.eye(2),
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

            dense_value = self.v_first.new_empty(
                self.v_rest.size(0) + 2, self.v_first.size(1)
            )
            nn.init.normal_(dense_value, mean=0.0, std=0.02)
            first_value = dense_value[0].clone()
            second_value = dense_value[1].clone()
            radius = torch.sqrt(
                first_value[0].square() + second_value[0].square()
            )
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first_value[0] / radius
            sine = second_value[0] / radius
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            dense_value[:2].copy_(rotation @ dense_value[:2].clone())
            self.v_first.copy_(dense_value[:1])
            self.v_second_tail.copy_(dense_value[1:2, 1:])
            self.v_rest.copy_(dense_value[2:])
            self.value_rotation.copy_(rotation)
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_head = self.head_basis @ second_head_coeff
        qk_weight = torch.cat((first_head, second_head), dim=0)
        return F.linear(
            x[..., :-1],
            torch.cat((qk_weight, self.v_weight), dim=0),
        )
=======
        second_head = self.head_basis @ second_head_coeff
        qk_weight = torch.cat((first_head, second_head), dim=0)
        v_weight = torch.cat(
            (
                self.v_first,
                F.pad(self.v_second_tail, (1, 0)),
                self.v_rest,
            ),
            dim=0,
        )
        return F.linear(
            x[..., :-1],
            torch.cat((qk_weight, v_weight), dim=0),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SingleRotationGaugeQKV):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SingleRotationGaugeQKV):
            module.reset_parameters()
        elif isinstance(module, CausalSelfAttention):
            with torch.no_grad():
                first_columns = module.proj.weight[:, :2].clone()
                module.proj.weight[:, :2].copy_(
                    first_columns @ module.qkv.value_rotation.transpose(0, 1)
                )
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE