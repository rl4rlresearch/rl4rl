MECHANISM: Alternative query-bias coordinate anchoring

HYPOTHESIS: Fixing `q_bias[4]` instead of `q_bias[3]` will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because coordinate-specific initialization effects previously allowed `fc1.bias[9]` to be removed successfully after removing the symmetric coordinate 8 failed.

INTENDED_EDIT: Retain learned query-bias coordinates 3, 5, 6, and 7 while reconstructing coordinates 0, 1, 2, and 4 as fixed zeros.

EVIDENCE: Fixing `q_bias[3]` collapsed accuracy to 14.80%, but the analogous hidden-bias sequence showed that failure of one symmetric coordinate need not imply failure of another: fixing `fc1.bias[8]` reached only 77.57%, whereas fixing coordinate 9 produced 100% accuracy.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
=======
        zero = self.q_bias_rest.new_zeros
        q_bias = torch.cat(
            (zero(3), self.q_bias_rest[:1], zero(1), self.q_bias_rest[1:])
        )
>>>>>>> REPLACE