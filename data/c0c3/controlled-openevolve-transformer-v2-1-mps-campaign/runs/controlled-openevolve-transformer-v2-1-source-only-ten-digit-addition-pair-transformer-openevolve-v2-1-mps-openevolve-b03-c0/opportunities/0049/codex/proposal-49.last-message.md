MECHANISM: LayerNorm-hyperplane anchoring of one query-projection weight

HYPOTHESIS: Anchoring the final input-weight coordinate of one query-projection neuron will reduce the model from 1,602 to 1,601 parameters while retaining at least 99% accuracy after 15,000 steps, because LayerNorm confines its input to a seven-dimensional affine hyperplane and the neuron retains an independent query bias.

INTENDED_EDIT: Store 191 of the 192 QKV weights, reconstruct coordinate 7 of the first query row as zero during forward passes, and add 1,000 low-rate refinement steps.

EVIDENCE: The analogous first-neuron `fc1` anchoring achieved 99.8% at 1,606 parameters, validating this LayerNorm-input redundancy; applying it to one query neuron tests the same mechanism in a different sublayer without repeating the failed second-`fc1` anchoring.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 16))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 16))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        qkv_weight = torch.cat(
            (
                self.qkv.weight[: d_model - 1],
                self.qkv.weight.new_zeros(1),
                self.qkv.weight[d_model - 1 :],
            )
        ).view(3 * d_model, d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=14000)
=======
    p.add_argument("--train-steps", type=int, default=15000)
>>>>>>> REPLACE