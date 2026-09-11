MECHANISM: Single-coordinate value-bias pruning with initialization preservation

HYPOTHESIS: Fixing one algebraically redundant value-projection bias coordinate to zero while preserving the baseline constructor RNG sequence will reduce parameters from 1,644 to 1,643 and retain at least 99% accuracy.

INTENDED_EDIT: Retain all query and key biases and seven of eight value biases; reconstruct the final value-bias coordinate as a fixed zero during the QKV projection.

EVIDENCE: The baseline reached 99.96%, whereas removing all eight value biases reached only 71.71%; pruning just one redundant coordinate is the smallest informative compression and avoids changing subsequent weight initialization.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the constructor's RNG consumption, then prune one value-bias coordinate.
        self.qkv.bias = nn.Parameter(self.qkv.bias.detach()[:-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
=======
        qkv = F.linear(x, self.qkv.weight) + F.pad(self.qkv.bias, (0, 1))
>>>>>>> REPLACE