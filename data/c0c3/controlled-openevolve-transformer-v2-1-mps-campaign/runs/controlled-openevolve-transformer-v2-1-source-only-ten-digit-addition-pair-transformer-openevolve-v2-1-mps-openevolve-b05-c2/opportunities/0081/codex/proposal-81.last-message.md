MECHANISM: Initialization-preserving value/output scale gauge

HYPOTHESIS: Fixing one nonzero first-head value-weight pivot at its fresh initialization will reduce the qualified model to 1506 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store the initialized `(first_value, 2)` pivot as a non-trainable checkpointed gauge anchor, train only the remaining row coordinates, and reconstruct the full value row during every forward pass.

EVIDENCE: The 1507-parameter nine-rotation model reached 99.91%, while additional value rotation and query-key constraints produced only 73.65% and 89.25%; value/output scaling is a distinct exact symmetry that preserves the initialized function without imposing another zero rotation.

<<<<<<< SEARCH
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
    """QKV map with eleven query-key, nine value rotations, and one value scale fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_value_weight = nn.Parameter(
            fixed_weight[value_start, 2:]
        )
=======
        self.register_buffer(
            "first_value_pivot",
            fixed_weight[value_start, 2].detach().clone(),
        )
        self.first_value_weight = nn.Parameter(
            fixed_weight[value_start, 3:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_value_weight.copy_(
                fixed_weight[value_start, 2:]
            )
=======
            self.first_value_pivot.copy_(
                fixed_weight[value_start, 2]
            )
            self.first_value_weight.copy_(
                fixed_weight[value_start, 3:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_value_row = F.pad(self.first_value_weight, (2, 0))
=======
        first_value_row = torch.cat(
            (
                self.first_value_weight.new_zeros(2),
                self.first_value_pivot.unsqueeze(0),
                self.first_value_weight,
            )
        )
>>>>>>> REPLACE