MECHANISM: Incremental softmax-invariant key-bias pruning

HYPOTHESIS: Fixing a second key-projection bias coordinate to zero will reduce parameters from 1,642 to 1,641 while retaining at least 99% accuracy, because key bias contributes only a row-wise constant to attention logits.

INTENDED_EDIT: Preserve constructor RNG consumption, retain six of eight key biases and seven of eight value biases, and reconstruct the omitted coordinates as fixed zeros during QKV projection.

EVIDENCE: Pruning one key-bias coordinate achieved 99.6% accuracy at 1,642 parameters; unlike the failed second value-bias pruning, another key coordinate is softmax-invariant, making this the smallest informative extension.

<<<<<<< SEARCH
        # Keep constructor RNG consumption, then prune one key- and one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[: 2 * d_model - 1], full_bias[2 * d_model : -1])).clone()
        )
=======
        # Keep constructor RNG consumption, then prune two key- and one value-bias coordinates.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[: 2 * d_model - 2], full_bias[2 * d_model : -1])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        kv_bias = self.qkv.bias[d_model:].view(2, d_model - 1)
        full_bias = torch.cat((q_bias, F.pad(kv_bias, (0, 1)).flatten()))
=======
        q_bias = self.qkv.bias[:d_model]
        key_bias = self.qkv.bias[d_model : 2 * d_model - 2]
        value_bias = self.qkv.bias[2 * d_model - 2 :]
        full_bias = torch.cat(
            (q_bias, F.pad(key_bias, (0, 2)), F.pad(value_bias, (0, 1)))
        )
>>>>>>> REPLACE