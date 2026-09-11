MECHANISM: Attention-LayerNorm offset absorption

HYPOTHESIS: Removing only the 8-parameter attention LayerNorm bias will reduce the model from 1636 to 1628 parameters while retaining at least 99% accuracy, because its query and value effects are absorbable by the retained biases and its key effect cancels under softmax.

INTENDED_EDIT: Disable the learned bias only in `ln1`, preserving `ln2` and every other successful architectural and training setting.

EVIDENCE: Removing both block LayerNorm biases reached only 91.12%, but that experiment confounded the attention and fragile MLP paths; the 1636-parameter model reached 99.84%, so isolating the theoretically redundant attention-side offset is the most informative smaller change.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE