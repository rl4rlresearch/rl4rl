MECHANISM: Alternative sixth-coordinate query-bias anchoring

HYPOTHESIS: Fixing `q_bias[6]` while retaining coordinates 3, 4, 5, and 7 will reduce the model from 1593 to 1592 parameters while maintaining at least 99% accuracy, because coordinate-specific training effects may make coordinate 6 removable even though removing coordinates 3, 4, or 5 failed.

INTENDED_EDIT: Store four learned query-bias coordinates and reconstruct coordinates 0, 1, 2, and 6 as fixed zeros.

EVIDENCE: The 1593-parameter design achieved 99.90% accuracy with coordinates 0 through 2 fixed; prior hidden-bias experiments showed that failure of one architecturally symmetric coordinate did not predict failure of another, motivating the next untested query-bias coordinate.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
=======
        zeros = self.q_bias_rest.new_zeros
        q_bias = torch.cat(
            (zeros(3), self.q_bias_rest[:3], zeros(1), self.q_bias_rest[3:])
        )
>>>>>>> REPLACE