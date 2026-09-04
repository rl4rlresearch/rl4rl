MECHANISM: Upstream-absorbed single query-bias quotient

HYPOTHESIS: Removing one raw query-bias scalar will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because the four retained `ln1` bias coordinates can generate that query offset through the dense query projection, while the remaining query/value biases compensate their coupled offsets.

INTENDED_EDIT: Omit the final query-bias coordinate, reconstruct it as zero in attention, and retain all value biases and the verified four-coordinate `ln1` bias.

EVIDENCE: Fixing `ln1` bias coordinate three nearly passed at 98.13%, showing this affine redundancy is promising but that retaining upstream adaptive bias dynamics matters; mean-tying that coordinate collapsed to 54.46%, motivating removal of a downstream query-bias degree instead.

<<<<<<< SEARCH
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
=======
        # Keep constructor RNG consumption, remove the softmax-null key bias,
        # and quotient the final query bias through the retained ln1 bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat(
                (full_bias[: d_model - 1], full_bias[2 * d_model :])
            ).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
=======
        stored_q_bias = self.qkv.bias[: d_model - 1]
        q_bias = F.pad(stored_q_bias, (0, 1))
        v_bias = self.qkv.bias[d_model - 1 :]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
>>>>>>> REPLACE