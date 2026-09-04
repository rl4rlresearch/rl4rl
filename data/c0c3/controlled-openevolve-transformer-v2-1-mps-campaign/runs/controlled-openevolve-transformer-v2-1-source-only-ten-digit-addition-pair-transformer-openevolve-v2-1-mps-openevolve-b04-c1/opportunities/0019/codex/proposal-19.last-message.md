MECHANISM: Sixth-coordinate query-bias ablation

HYPOTHESIS: Learning two of eight query-bias coordinates will reduce the model from 1610 to 1609 parameters while retaining at least 99% accuracy, because every prior query-bias reduction through three learned coordinates retained at least 99.98% accuracy while preserving the necessary three-coordinate value bias.

INTENDED_EDIT: Store two learned query-bias coordinates and pad the remaining six coordinates with zeros during attention.

EVIDENCE: The three-coordinate query-bias design achieved 99.98% accuracy with 1610 parameters, and successive reductions from eight through three coordinates all exceeded 99.99% except the latest still-strong result; continuing this ablation is the most informative test of its capacity boundary.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + F.pad(self.q_bias, (0, 5))
=======
        q = q + F.pad(self.q_bias, (0, 6))
>>>>>>> REPLACE