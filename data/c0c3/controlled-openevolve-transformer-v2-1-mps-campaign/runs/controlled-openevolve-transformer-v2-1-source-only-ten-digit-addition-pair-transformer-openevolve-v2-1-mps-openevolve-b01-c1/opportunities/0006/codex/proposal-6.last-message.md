MECHANISM: Two-coordinate value-bias pruning with initialization preservation

HYPOTHESIS: Fixing a second algebraically redundant value-projection bias coordinate to zero will reduce parameters from 1,643 to 1,642 while retaining at least 99% accuracy.

INTENDED_EDIT: Preserve constructor RNG consumption, retain six of eight learned value biases, and reconstruct the final two value-bias coordinates as fixed zeros during projection.

EVIDENCE: Pruning one value-bias coordinate with unchanged initialization achieved 99.96% accuracy at 1,643 parameters; extending that successful compression by one coordinate is the smallest informative next change.

<<<<<<< SEARCH
        # Keep the constructor's RNG consumption, then prune one value-bias coordinate.
        self.qkv.bias = nn.Parameter(self.qkv.bias.detach()[:-1].clone())
=======
        # Keep the constructor's RNG consumption, then prune two value-bias coordinates.
        self.qkv.bias = nn.Parameter(self.qkv.bias.detach()[:-2].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + F.pad(self.qkv.bias, (0, 1))
=======
        qkv = F.linear(x, self.qkv.weight) + F.pad(self.qkv.bias, (0, 2))
>>>>>>> REPLACE