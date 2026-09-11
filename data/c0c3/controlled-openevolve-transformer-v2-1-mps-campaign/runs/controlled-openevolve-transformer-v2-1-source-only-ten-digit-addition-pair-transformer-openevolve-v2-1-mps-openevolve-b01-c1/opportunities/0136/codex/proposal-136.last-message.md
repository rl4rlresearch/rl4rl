MECHANISM: Single-coordinate value-bias/projection-bias quotient

HYPOTHESIS: Fixing the final attention value-bias coordinate at zero will reduce the model from 1,522 to 1,521 parameters while retaining at least 99% accuracy, because softmax-normalized attention makes value bias position-independent and its projected contribution can be represented by the trainable attention projection bias.

INTENDED_EDIT: Retain all query-bias coordinates and the first seven value-bias coordinates, reconstructing the final value-bias coordinate as zero during the forward pass.

EVIDENCE: Removing the next `ln1` bias coordinate collapsed accuracy to 72.53%, motivating a different one-parameter reduction. Unlike that optimization-sensitive LayerNorm restriction, this targets a direct redundancy between a value-bias coordinate and the existing learned projection bias.

<<<<<<< SEARCH
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
=======
        # Keep constructor RNG consumption, remove the softmax-null key bias,
        # and quotient the final value-bias coordinate through proj.bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat(
                (full_bias[:d_model], full_bias[2 * d_model : -1])
            ).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
=======
        q_bias = self.qkv.bias[:d_model]
        retained_v_bias = self.qkv.bias[d_model:]
        v_bias = F.pad(retained_v_bias, (0, 1))
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
>>>>>>> REPLACE