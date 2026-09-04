MECHANISM: Third query-bias coordinate anchoring

HYPOTHESIS: Fixing `q_bias[2]` at zero will reduce the model from 1594 to 1593 parameters while retaining at least 99% accuracy, because fixing coordinates 0 and 1 consecutively preserved 99.97% accuracy and all query-bias coordinates share zero initialization.

INTENDED_EDIT: Replace the six learned query-bias coordinates with five, reconstructing coordinates 0 through 2 as fixed zeros.

EVIDENCE: The 1594-parameter design achieved 99.97% accuracy after fixing `q_bias[1]`; extending this successful query-bias constraint is the most informative next capacity test.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(2), self.q_bias_rest))
=======
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
>>>>>>> REPLACE