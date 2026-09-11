MECHANISM: Cross-head value-bias coordinate tying

HYPOTHESIS: Tying value-bias coordinates 1 and `head_dim + 1` will reduce the model to 1,616 parameters while retaining at least 99% accuracy, because both coordinates represent constant per-head offsets absorbable by the output-projection bias, while a learned symmetric tie preserves more optimization freedom than fixing another coordinate to zero.

INTENDED_EDIT: Store 20 QKV bias parameters and reconstruct the first learned value-bias coordinate of the second head from the corresponding coordinate of the first head, retaining the validated zero key coordinate and balanced zero value coordinates.

EVIDENCE: Removing value coordinates 0 and `head_dim` symmetrically across heads achieved 99.97% at 1,617 parameters, whereas concentrating two removed value coordinates in the first head achieved only 96.11%; this motivates a cross-head tie that preserves the successful balanced structure.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 4))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[2 * d_model - 1 : 2 * d_model + self.head_dim - 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
=======
                self.qkv.bias[2 * d_model - 1 : 2 * d_model + self.head_dim - 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 1 : 2 * d_model],
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
>>>>>>> REPLACE