MECHANISM: One-coordinate value-bias gauge fixing

HYPOTHESIS: Fixing one value-bias coordinate at zero will reduce the model from 1636 to 1635 parameters while retaining at least 99% accuracy, because the full output-projection bias preserves additive output freedom and seven value-bias coordinates retain nearly all of the optimization benefit lost when the entire value bias was removed.

INTENDED_EDIT: Store seven learned value-bias coordinates and append one fixed zero before applying the bias, leaving initialization and every other architectural and training setting unchanged.

EVIDENCE: Removing all eight value-bias parameters contributed to a collapse to 52.73%, while the full value bias achieved 99.84%; removing only one coordinate is the smallest titration of that optimization-sensitive redundancy.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v = v + self.v_bias
=======
        q = q + self.q_bias
        v = v + F.pad(self.v_bias, (0, 1))
>>>>>>> REPLACE