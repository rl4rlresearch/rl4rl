MECHANISM: Bias-free QKV projection

HYPOTHESIS: Removing the 24 QKV bias parameters will reduce the model from 1,644 to 1,620 parameters while retaining at least 99% accuracy because LayerNorm already provides learned channel offsets without reducing attention or MLP width.

INTENDED_EDIT: Disable the bias on the combined query, key, and value projection while leaving all model widths and training settings unchanged.

EVIDENCE: The 1,644-parameter baseline reached 99.96%, whereas reducing `d_ff` to 10 or 8 failed; this motivates preserving feed-forward capacity and instead testing a redundant affine component.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE