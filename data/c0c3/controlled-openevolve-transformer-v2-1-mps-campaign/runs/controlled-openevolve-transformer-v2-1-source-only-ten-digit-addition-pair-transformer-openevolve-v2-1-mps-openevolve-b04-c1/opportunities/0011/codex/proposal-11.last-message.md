MECHANISM: Four-coordinate value-bias ablation

HYPOTHESIS: Reducing the learned value bias from five to four coordinates will lower parameters from 1617 to 1616 while retaining at least 99% accuracy, because the three-coordinate ablation achieved 99.91% and remains far above the 97.53% observed only when all eight coordinates were removed.

INTENDED_EDIT: Store four learned value-bias coordinates and pad the remaining four coordinates with zeros during attention.

EVIDENCE: The current five-coordinate value bias reached 99.91% with 1617 parameters, while six coordinates reached 99.58%; continuing the one-coordinate ablation is the most direct test of the accuracy boundary.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 4))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v = v + F.pad(self.v_bias, (0, 3))
=======
        q = q + self.q_bias
        v = v + F.pad(self.v_bias, (0, 4))
>>>>>>> REPLACE