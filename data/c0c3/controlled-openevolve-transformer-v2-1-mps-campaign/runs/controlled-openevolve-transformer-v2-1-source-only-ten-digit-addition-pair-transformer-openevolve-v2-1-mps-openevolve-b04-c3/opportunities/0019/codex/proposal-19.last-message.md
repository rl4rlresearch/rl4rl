MECHANISM: Joint key/value bias quotient in causal attention

HYPOTHESIS: Removing all key-projection biases and the value-projection bias will retain at least 99% accuracy after 50,000 steps with 1,580 learned parameters, because key bias is softmax-invariant and value bias can be absorbed exactly by the retained attention output bias.

INTENDED_EDIT: Store only the eight query-bias parameters, reconstruct fixed-zero key and value biases in the fused QKV projection, and increase default training from 25,000 to 50,000 steps.

EVIDENCE: The 1,588-parameter design removing every key bias achieved 99.92% accuracy at 44,999 steps. The next eight parameters are functionally redundant: attention weights sum to one, so value bias adds the same vector at every position, which the learned output-projection bias can represent directly.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Four key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 4))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(4),
                self.qkv.bias[d_model:],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=25000)
=======
    p.add_argument("--train-steps", type=int, default=50000)
>>>>>>> REPLACE