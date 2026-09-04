MECHANISM: Stop-gradient key-bias group sharing

HYPOTHESIS: Replacing the remaining independent key-bias group with detached views of the surviving learned key scalar and adding 1,000 low-rate steps will reduce the model from 1,610 to 1,609 parameters while retaining at least 99% accuracy, because the reconstructed offsets remain softmax-invariant without adding the failed group’s gradients to the surviving scalar.

INTENDED_EDIT: Store 13 QKV bias parameters, reconstruct the removed key-bias group from detached views of the surviving key scalar, shift value-bias slices to preserve their mapping, and train for 7,000 steps on the existing 5,000-step cosine schedule.

EVIDENCE: The current 1,610-parameter key-bias merge reached 99.88% after 1,000 low-rate refinement steps, while earlier direct key-group merges failed; stop-gradient sharing tests whether preserving the successful scalar’s existing gradient paths avoids that optimization failure.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 10))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 11))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + self.head_dim - 2],
                self.qkv.bias[d_model + self.head_dim - 3 : d_model + self.head_dim - 2],
                self.qkv.bias[d_model + self.head_dim - 3 : d_model + self.head_dim - 2],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model + self.head_dim - 3 : d_model + self.head_dim - 2],
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 6 : 2 * d_model + self.head_dim - 9],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 6 : 2 * d_model - 4],
                self.qkv.bias[2 * d_model + self.head_dim - 9 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 7 : 2 * d_model + self.head_dim - 10],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 7 : 2 * d_model - 5],
                self.qkv.bias[2 * d_model + self.head_dim - 10 :],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=6000)
=======
    p.add_argument("--train-steps", type=int, default=7000)
>>>>>>> REPLACE