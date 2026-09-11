MECHANISM: Fourth query-bias coordinate anchoring

HYPOTHESIS: Fixing `q_bias[3]` at zero will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because coordinates 0 through 2 were removed consecutively without violating the accuracy requirement and all query-bias coordinates share zero initialization.

INTENDED_EDIT: Replace the five learned query-bias coordinates with four, reconstructing coordinates 0 through 3 as fixed zeros.

EVIDENCE: The 1593-parameter design achieved 99.90% accuracy after fixing `q_bias[2]`; extending this three-step successful query-bias constraint is the most informative next capacity test.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
=======
        q_bias = torch.cat((self.q_bias_rest.new_zeros(4), self.q_bias_rest))
>>>>>>> REPLACE