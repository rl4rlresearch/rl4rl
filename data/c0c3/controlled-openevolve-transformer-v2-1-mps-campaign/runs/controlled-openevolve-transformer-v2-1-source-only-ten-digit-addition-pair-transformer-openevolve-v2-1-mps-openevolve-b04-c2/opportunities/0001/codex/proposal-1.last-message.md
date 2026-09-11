MECHANISM: Bias-free transformer projections

HYPOTHESIS: Removing redundant projection biases will reduce parameters from 1,644 to 1,592 while preserving at least 99% accuracy.

INTENDED_EDIT: Disable biases in all attention and MLP linear layers without changing model width, depth, training, or decoding.

EVIDENCE: The current 1,644-parameter model reaches 99.96% accuracy, providing enough margin for a conservative 52-parameter reduction that preserves every learned weight matrix.

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