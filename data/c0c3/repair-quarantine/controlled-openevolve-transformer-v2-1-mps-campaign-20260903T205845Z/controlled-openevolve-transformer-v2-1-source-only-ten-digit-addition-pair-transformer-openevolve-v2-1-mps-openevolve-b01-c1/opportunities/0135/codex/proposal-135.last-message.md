MECHANISM: Parameter-free radial LayerNorm scale

HYPOTHESIS: Deriving `ln2` coordinate-six scale from the corresponding `fc1` column norm will reduce the model to 1,524 parameters while retaining at least 99% accuracy, because it preserves the verified initialization and supplies adaptive radial training dynamics without an independent scale parameter.

INTENDED_EDIT: Remove the learned coordinate-six `ln2` scale, reconstruct it from the square root of the normalized `fc1` coordinate-six column norm, and retain the column’s initialized norm as a non-trainable buffer.

EVIDENCE: Fixing coordinate six at one collapsed accuracy to 66.5%, while tying it to the other learned scales reached 97.73%; this shows adaptive scale dynamics help, but coupling them to load-bearing LayerNorm coordinates remains too restrictive.

<<<<<<< SEARCH
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())
        self.register_buffer("initial_sixth_coordinate_norm", torch.ones(()))

    def sixth_coordinate_values(self) -> torch.Tensor:
        return torch.cat(
            (
                self.first_three_rows[:, 6],
                self.fourth_row[5:6],
                self.fifth_row[5:6],
                self.weight[:, 5],
                self.seventh_row[5:6],
                self.eighth_row[5:6],
                self.last_four_rows[:, 6],
            )
        )

    def sixth_coordinate_scale(self) -> torch.Tensor:
        norm_ratio = (
            self.sixth_coordinate_values().norm()
            / self.initial_sixth_coordinate_norm
        )
        return torch.sqrt(norm_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and five scales fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (full_weight[:1], full_weight[3:4], full_weight[6:-1])
            ).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:2],
                self.weight.new_ones(2),
                self.weight[2:],
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
=======
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and a parameter-free adaptive sixth scale."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat((full_weight[:1], full_weight[3:4])).clone()
        )
        self.bias = None

    def forward(
        self, x: torch.Tensor, sixth_coordinate_scale: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:2],
                self.weight.new_ones(2),
                sixth_coordinate_scale.unsqueeze(0),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        sixth_coordinate_scale = self.mlp.fc1.sixth_coordinate_scale()
        x = x + self.mlp(self.ln2(x, sixth_coordinate_scale))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.last_four_rows.copy_(
                    full[-4:, :-1] - full[-4:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
=======
                module.last_four_rows.copy_(
                    full[-4:, :-1] - full[-4:, -1].unsqueeze(1)
                )
                module.initial_sixth_coordinate_norm.copy_(
                    module.sixth_coordinate_values().norm()
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE