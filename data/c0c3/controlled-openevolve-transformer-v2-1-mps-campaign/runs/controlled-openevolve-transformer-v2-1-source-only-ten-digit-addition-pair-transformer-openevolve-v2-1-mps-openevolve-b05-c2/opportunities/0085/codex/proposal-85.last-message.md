MECHANISM: Final second-head query-bias coordinate anchoring

HYPOTHESIS: Fixing the remaining untested `q_bias[7]` will reduce the qualified 1506-parameter model to 1505 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct query-bias coordinates 0, 1, 2, 4, 6, and 7 as zero, retaining learned biases only at coordinates 3 and 5.

EVIDENCE: Fixing `q_bias[6]` retained 99.78% accuracy at 1506 parameters; `q_bias[7]` is the only query-bias coordinate not already qualified or shown harmful, making it the most informative adjacent reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 2:],
            )
=======
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 2:],
                self.q_bias.new_zeros(1),
            )
>>>>>>> REPLACE