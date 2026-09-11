MECHANISM: Alternative seventh-coordinate query-bias anchoring

HYPOTHESIS: Fixing `q_bias[7]` while retaining coordinates 3 through 6 will reduce the model from 1593 to 1592 parameters while maintaining at least 99% accuracy, because coordinate-specific training effects may make the sole untested query-bias coordinate removable even though removing coordinates 3 through 6 failed.

INTENDED_EDIT: Store four learned query-bias coordinates and reconstruct coordinates 0, 1, 2, and 7 as fixed zeros.

EVIDENCE: Fixing query-bias coordinates 0 through 2 preserved 99.90% accuracy, while removing coordinates 3, 4, 5, or 6 individually failed; coordinate 7 is the only remaining untested alternative, and earlier hidden-bias results showed that failures of symmetric coordinates did not rule out success for another coordinate.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
=======
        q_bias = torch.cat(
            (self.q_bias_rest.new_zeros(3), self.q_bias_rest, self.q_bias_rest.new_zeros(1))
        )
>>>>>>> REPLACE