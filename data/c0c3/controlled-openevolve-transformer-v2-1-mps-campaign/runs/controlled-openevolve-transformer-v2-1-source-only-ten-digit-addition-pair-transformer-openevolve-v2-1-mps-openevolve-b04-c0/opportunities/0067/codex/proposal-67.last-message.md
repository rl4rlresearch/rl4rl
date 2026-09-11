MECHANISM: Alternative fifth-coordinate query-bias anchoring

HYPOTHESIS: Fixing `q_bias[5]` while retaining coordinates 3, 4, 6, and 7 will reduce the model from 1593 to 1592 parameters while maintaining at least 99% accuracy, because coordinate-specific training effects may make coordinate 5 removable even though removing coordinates 3 or 4 failed.

INTENDED_EDIT: Store four learned query-bias coordinates and reconstruct coordinates 0, 1, 2, and 5 as fixed zeros.

EVIDENCE: Fixing query-bias coordinates 0 through 2 preserved 99.90% accuracy, while alternative removals of coordinates 3 and 4 failed; earlier hidden-bias results showed that one symmetric coordinate can fail badly while another succeeds, motivating the next untested coordinate.

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
            (zeros(3), self.q_bias_rest[:2], zeros(1), self.q_bias_rest[2:])
        )
>>>>>>> REPLACE