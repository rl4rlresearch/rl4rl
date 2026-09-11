MECHANISM: Bias-free transformer projections

HYPOTHESIS: Removing linear-layer biases while preserving the full 8-dimensional positional embeddings will retain at least 99% accuracy because pre-LayerNorm affine terms and residual paths provide learned offsets without reducing attention or MLP rank.

INTENDED_EDIT: Disable biases in both attention projections and both MLP projections, reducing learned parameters while leaving model width, positional capacity, and training unchanged.

EVIDENCE: The rank-4 positional bottleneck reduced accuracy from 99.96% to 50.39%, showing that positional capacity should be preserved; this patch instead removes projection offsets that are typically redundant in a pre-LayerNorm transformer.

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