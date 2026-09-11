MECHANISM: Second query-bias gauge anchoring

HYPOTHESIS: Fixing `q_bias[1]` at zero will reduce the model from 1595 to 1594 parameters while retaining at least 99% accuracy, because coordinate 0 was removed successfully and the zero-initialized query-bias coordinates are architecturally symmetric.

INTENDED_EDIT: Replace the seven learned query-bias coordinates with six, reconstructing coordinates 0 and 1 as fixed zeros while preserving initialization behavior.

EVIDENCE: Fixing `q_bias[0]` reduced the model to 1595 parameters with 99.56% accuracy; extending that successful attention-side constraint is the most direct next capacity test.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 1))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 2))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(1), self.q_bias_rest))
=======
        q_bias = torch.cat((self.q_bias_rest.new_zeros(2), self.q_bias_rest))
>>>>>>> REPLACE