MECHANISM: Bias-free learned projections

HYPOTHESIS: Removing all linear-layer biases will reduce parameters from 1644 to 1592 while retaining at least 99% accuracy because it preserves the verified 8-wide residual stream, two-head attention, and 12-wide MLP.

INTENDED_EDIT: Disable biases in the attention and MLP linear projections without changing model width, training, or decoding.

EVIDENCE: The 8-wide baseline achieved 0.9996 accuracy, whereas reducing `d_model` to 6 collapsed accuracy to 0.0001; this motivates preserving residual capacity and removing less-essential projection biases instead.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
>>>>>>> REPLACE