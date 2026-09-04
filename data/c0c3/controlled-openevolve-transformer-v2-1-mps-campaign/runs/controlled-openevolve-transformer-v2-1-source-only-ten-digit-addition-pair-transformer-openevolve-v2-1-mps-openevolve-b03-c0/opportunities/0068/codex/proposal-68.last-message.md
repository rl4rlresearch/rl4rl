MECHANISM: Trajectory-preserving column-1 MLP output gauge

HYPOTHESIS: Gauge-anchoring initialized `fc2` column 1 will reduce the model from 1,589 to 1,588 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Split `fc2` column 1 into seven relative coordinates plus a zero anchor after initialization, reconstruct the original column order in forward passes, and optimize it with `GaugeAdamW`.

EVIDENCE: Post-initialization gauges retained 99.95% at column 0 and 99.93% at column 3, while only column 6 failed; testing column 1 applies the validated exact symmetry in an untested region away from the failed column without perturbing the initialization draw stream.

<<<<<<< SEARCH
        self.register_parameter("fc2_col_0", None)
        self.register_parameter("fc2_col_mid", None)
=======
        self.register_parameter("fc2_col_0", None)
        self.register_parameter("fc2_col_1", None)
        self.register_parameter("fc2_col_mid", None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc2_col_0 = torch.cat(
            (self.fc2_col_0, self.fc2_col_0.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_mid = torch.cat(
=======
        fc2_col_0 = torch.cat(
            (self.fc2_col_0, self.fc2_col_0.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_1 = torch.cat(
            (self.fc2_col_1, self.fc2_col_1.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_mid = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                fc2_col_0,
                self.fc2.weight[:, :2],
                fc2_col_mid,
                self.fc2.weight[:, 2:],
=======
                fc2_col_0,
                fc2_col_1,
                self.fc2.weight[:, :1],
                fc2_col_mid,
                self.fc2.weight[:, 1:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_fc2_col_0 = block.mlp.fc2.weight[:, 0].detach().clone()
                full_fc2_col_mid = block.mlp.fc2.weight[:, 3].detach().clone()
=======
                full_fc2_col_0 = block.mlp.fc2.weight[:, 0].detach().clone()
                full_fc2_col_1 = block.mlp.fc2.weight[:, 1].detach().clone()
                full_fc2_col_mid = block.mlp.fc2.weight[:, 3].detach().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                        block.mlp.fc2.weight[:, 1:3],
                        block.mlp.fc2.weight[:, 4:-5],
=======
                        block.mlp.fc2.weight[:, 2:3],
                        block.mlp.fc2.weight[:, 4:-5],
>>>>>>> REPLACE

<<<<<<< SEARCH
                block.mlp.fc2_col_0 = nn.Parameter(
                    full_fc2_col_0[:-1].clone()
                )
                full_fc2_col_mid.sub_(full_fc2_col_mid[-1].clone())
=======
                block.mlp.fc2_col_0 = nn.Parameter(
                    full_fc2_col_0[:-1].clone()
                )
                full_fc2_col_1.sub_(full_fc2_col_1[-1].clone())
                block.mlp.fc2_col_1 = nn.Parameter(
                    full_fc2_col_1[:-1].clone()
                )
                full_fc2_col_mid.sub_(full_fc2_col_mid[-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        block.mlp.fc2_col_0 for block in model.blocks
    ] + [
        block.mlp.fc2_col_mid for block in model.blocks
=======
    ] + [
        block.mlp.fc2_col_0 for block in model.blocks
    ] + [
        block.mlp.fc2_col_1 for block in model.blocks
    ] + [
        block.mlp.fc2_col_mid for block in model.blocks
>>>>>>> REPLACE