MECHANISM: Single-coordinate value-bias ablation

HYPOTHESIS: Removing one of eight value-bias coordinates will reduce parameters from 1620 to 1619 while retaining at least 99% accuracy, because the full removal narrowly missed at 97.53% and the retained seven coordinates preserve most of its optimization benefit while `proj.bias` can absorb the missing constant offset.

INTENDED_EDIT: Store seven learned value-bias coordinates and pad the eighth with zero during attention.

EVIDENCE: Removing all eight value-bias parameters produced 97.53% accuracy, whereas the current 1620-parameter model achieved 99.98%; this motivates a minimal one-coordinate ablation rather than another full bias removal.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + self.v_bias
=======
        v = v + F.pad(self.v_bias, (0, 1))
>>>>>>> REPLACE