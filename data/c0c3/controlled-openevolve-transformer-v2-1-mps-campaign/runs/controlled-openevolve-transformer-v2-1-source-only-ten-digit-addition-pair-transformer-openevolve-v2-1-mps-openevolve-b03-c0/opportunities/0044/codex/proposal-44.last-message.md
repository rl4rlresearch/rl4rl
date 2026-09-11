MECHANISM: Incremental LayerNorm-hyperplane anchoring

HYPOTHESIS: Anchoring the final input-weight coordinate of a second `fc1` neuron will reduce the model from 1,606 to 1,605 parameters while retaining at least 99% accuracy after 11,000 steps, because every MLP neuron receives inputs confined by LayerNorm to the same seven-dimensional affine hyperplane and keeps an independent learned bias.

INTENDED_EDIT: Store 94 of the 96 `fc1` weights, reconstruct coordinate 7 of the final two neuron rows as zero, and add 1,000 low-learning-rate refinement steps.

EVIDENCE: Anchoring this coordinate for one `fc1` neuron achieved 99.8% at 1,606 parameters after 10,000 steps, directly validating the same per-neuron redundancy and training strategy.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 2))
        self.fc2 = nn.Linear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc1_weight = torch.cat(
            (self.fc1.weight, self.fc1.weight.new_zeros(1))
        ).view(self.fc1.out_features, self.fc1.in_features)
=======
        fc1_weight = torch.cat(
            (
                self.fc1.weight[: self.fc1.weight.numel() - 7],
                self.fc1.weight.new_zeros(1),
                self.fc1.weight[self.fc1.weight.numel() - 7 :],
                self.fc1.weight.new_zeros(1),
            )
        ).view(self.fc1.out_features, self.fc1.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=10000)
=======
    p.add_argument("--train-steps", type=int, default=11000)
>>>>>>> REPLACE