MECHANISM: Softmax-null key-bias elimination

HYPOTHESIS: Removing all seven remaining learned key-bias coordinates will reduce parameters from 1,593 to 1,586 while retaining at least 99% accuracy, because each head’s position-independent key bias adds the same scalar to every unmasked attention logit and is canceled exactly by softmax.

INTENDED_EDIT: Retain learned query biases and the seven-coordinate value bias, but reconstruct the entire key bias as zeros while preserving the original linear constructor and initialization behavior.

EVIDENCE: The current design achieved 99.88% at 1,593 parameters, whereas further LayerNorm-coordinate pruning removed useful capacity and failed; key bias elimination instead removes an exact attention-softmax invariance without narrowing learned queries, keys, values, or attention distributions.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then prune one key- and one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[: 2 * d_model - 1], full_bias[2 * d_model : -1])).clone()
        )
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and prune one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        kv_bias = self.qkv.bias[d_model:].view(2, d_model - 1)
        full_bias = torch.cat((q_bias, F.pad(kv_bias, (0, 1)).flatten()))
=======
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), F.pad(v_bias, (0, 1)))
        )
>>>>>>> REPLACE