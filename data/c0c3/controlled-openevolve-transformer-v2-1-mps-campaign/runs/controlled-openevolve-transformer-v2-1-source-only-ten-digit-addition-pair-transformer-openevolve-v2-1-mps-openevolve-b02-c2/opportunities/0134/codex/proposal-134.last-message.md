MECHANISM: Complete pre-MLP LayerNorm-bias absorption

HYPOTHESIS: Removing the final `ln2` bias parameter will yield 1,576 parameters with at least 99% accuracy because the downstream learned `fc1` bias exactly absorbs every `ln2` bias contribution; reducing periodic validation frequency should allow the previously timed-out design to finish.

INTENDED_EDIT: Support a parameter-free LayerNorm bias when all coordinates are fixed, set `ln2` to fix all eight bias coordinates, and reduce validation overhead without changing the 5,000-step optimization trajectory.

EVIDENCE: The 1,577-parameter current design achieved 99.81%, and fixing the seventh `ln2` bias coordinate independently achieved 99.97%; the prior complete-absorption attempt timed out rather than failing accuracy.

<<<<<<< SEARCH
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        if fixed_coordinates == self.normalized_shape[0]:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                layer_norm.bias[:-fixed_coordinates].detach().clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.fixed_coordinates))
        )
        return F.layer_norm(
=======
        if self.bias is None:
            full_bias = full_weight.new_zeros(self.normalized_shape)
        else:
            full_bias = torch.cat(
                (self.bias, self.bias.new_zeros(self.fixed_coordinates))
            )
        return F.layer_norm(
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=7,
            fixed_weight_coordinates=2,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=8,
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=500)
>>>>>>> REPLACE