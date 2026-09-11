MECHANISM: Incremental value/output-bias quotient

HYPOTHESIS: Fixing a second value-bias coordinate to zero will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because value bias is functionally absorbable into the attention output bias and the successful current model already fixes one such coordinate.

INTENDED_EDIT: Store six rather than seven value-bias coordinates and reconstruct the final two coordinates as zero.

EVIDENCE: The 1628-parameter model achieved 99.40% while fixing one value-bias coordinate, whereas every attempted additional first-head key gauge failed; titrating the distinct value/output-bias redundancy is therefore the most informative next reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = nn.Linear(d_model, d_model, bias=False)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 2))
        self.proj = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v = v + F.pad(self.v_bias, (0, 1))
=======
        q = q + self.q_bias
        v = v + F.pad(self.v_bias, (0, 2))
>>>>>>> REPLACE