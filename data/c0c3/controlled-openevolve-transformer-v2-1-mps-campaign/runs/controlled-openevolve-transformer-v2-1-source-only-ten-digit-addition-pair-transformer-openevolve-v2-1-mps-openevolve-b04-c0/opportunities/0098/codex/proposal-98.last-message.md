MECHANISM: Independent-head query-bias gauge anchor

HYPOTHESIS: Fixing `q_bias[4]` at zero will reduce the model to 1576 parameters while retaining at least 99% accuracy, because head 0 already tolerates three fixed query-bias coordinates and head 1 retains three learned bias coordinates after this first analogous anchor.

INTENDED_EDIT: Store four learned query-bias values and reconstruct coordinates 0, 1, 2, and 4 as zero without disturbing the zero-bias initialization.

EVIDENCE: The verified 1577-parameter design achieves 99.13% accuracy with query-bias coordinates 0 through 2 already fixed at zero; applying the first bias anchor to the independent second head tests the same successful redundancy while avoiding the failed fourth head-0 scale anchor.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 4))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        q = q + q_bias
=======
        zero = self.q_bias_rest.new_zeros
        q_bias = torch.cat((zero(3), self.q_bias_rest[:1], zero(1), self.q_bias_rest[1:]))
        q = q + q_bias
>>>>>>> REPLACE