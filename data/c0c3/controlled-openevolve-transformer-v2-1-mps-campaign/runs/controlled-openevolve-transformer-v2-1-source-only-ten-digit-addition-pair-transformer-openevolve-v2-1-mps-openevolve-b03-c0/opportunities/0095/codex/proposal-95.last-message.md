MECHANISM: Fused final MLP output-column shift gauge

HYPOTHESIS: Anchoring the sole remaining full `fc2` column inside its existing ordinary-AdamW parameter will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps and avoiding an additional per-step gauge optimizer.

INTENDED_EDIT: Store MLP column 6 as seven relative coordinates in the existing `fc2.weight` parameter, reconstruct its zero-anchored eighth coordinate during the forward pass, and leave it in the ordinary optimizer.

EVIDENCE: The verified 1,573-parameter design already gauge-anchors the other eleven `fc2` columns at 99.98% accuracy; the prior final-column attempt timed out without adverse accuracy evidence, motivating the same exact symmetry with no additional custom optimizer.

<<<<<<< SEARCH
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_col_0 = torch.cat(
=======
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_col_6 = torch.cat(
            (self.fc2.weight, self.fc2.weight.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_0 = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.fc2_col_5_abs,
                self.fc2.weight,
                fc2_col_5,
=======
                self.fc2_col_5_abs,
                fc2_col_6,
                fc2_col_5,
>>>>>>> REPLACE

<<<<<<< SEARCH
                base_fc2_weight = block.mlp.fc2.weight[:, 6:-5].detach().clone()
                block.mlp.fc2.weight = nn.Parameter(base_fc2_weight)
=======
                full_fc2_col_6 = block.mlp.fc2.weight[:, 6].detach().clone()
                full_fc2_col_6.sub_(full_fc2_col_6[-1].clone())
                block.mlp.fc2.weight = nn.Parameter(
                    full_fc2_col_6[:-1].clone()
                )
>>>>>>> REPLACE