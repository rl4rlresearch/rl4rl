MECHANISM: Cross-projection per-head bias-capacity balancing

HYPOTHESIS: Omitting three value-bias coordinates from head 0 and all four from head 1 will produce a 1,630-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend `CompactQKV` from a 3:3 to a 3:4 value-bias omission layout, balancing total key-plus-value omissions at seven per head while preserving all QKV weights and baseline initialization draws.

EVIDENCE: The balanced 2:3 layout achieved 99.99% at 1,632 parameters, and the subsequent 3:3 layout achieved 99.6% at 1,631; assigning the next omission to head 1 restores the successful cross-projection balance.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and three value biases per head."""
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and value biases in a 3:4 layout."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                linear.bias[self.value_start + self.head_dim + 3 :],
=======
                linear.bias[self.value_start + self.head_dim + 4 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.bias.new_zeros(3),
                self.bias[first_value_end:],
=======
                self.bias.new_zeros(4),
                self.bias[first_value_end:],
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and three value biases from each head.
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and value biases in a 3:4 per-head layout.
>>>>>>> REPLACE