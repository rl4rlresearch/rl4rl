MECHANISM: Single-coordinate key-bias pruning with initialization preservation

HYPOTHESIS: Fixing one softmax-invariant key-bias coordinate to zero will reduce parameters from 1,643 to 1,642 while retaining at least 99% accuracy.

INTENDED_EDIT: Preserve constructor RNG consumption, retain seven of eight key biases and seven of eight value biases, and reconstruct their final coordinates as fixed zeros during QKV projection.

EVIDENCE: The one-coordinate value-bias reduction achieved 99.96%, while removing all eight key biases achieved only 71.77%; this tests the smallest incremental key-bias reduction without repeating the failed wholesale removal.

<<<<<<< SEARCH
        # Keep the constructor's RNG consumption, then prune one value-bias coordinate.
        self.qkv.bias = nn.Parameter(self.qkv.bias.detach()[:-1].clone())
=======
        # Keep constructor RNG consumption, then prune one key- and one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[: 2 * d_model - 1], full_bias[2 * d_model : -1])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + F.pad(self.qkv.bias, (0, 1))
=======
        q_bias = self.qkv.bias[:d_model]
        kv_bias = self.qkv.bias[d_model:].view(2, d_model - 1)
        full_bias = torch.cat((q_bias, F.pad(kv_bias, (0, 1)).flatten()))
        qkv = F.linear(x, self.qkv.weight) + full_bias
>>>>>>> REPLACE