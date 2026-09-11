MECHANISM: Single-coordinate query-bias ablation

HYPOTHESIS: Learning seven of eight query-bias coordinates will reduce the model from 1615 to 1614 parameters while retaining at least 99% accuracy, because it preserves the three-coordinate value bias that achieved 100% accuracy and makes only a one-coordinate change to the otherwise verified architecture.

INTENDED_EDIT: Store seven learned query-bias coordinates and pad the remaining coordinate with zero during attention.

EVIDENCE: Reducing the value bias from three coordinates to two collapsed accuracy from 100% to 55.14%, identifying the three-coordinate value bias as a boundary worth preserving; a one-coordinate ablation of the still-intact eight-coordinate query bias is therefore the most informative alternative reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v = v + F.pad(self.v_bias, (0, 5))
=======
        q = q + F.pad(self.q_bias, (0, 1))
        v = v + F.pad(self.v_bias, (0, 5))
>>>>>>> REPLACE