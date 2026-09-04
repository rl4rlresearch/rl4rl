MECHANISM: Residual-output bias pruning

HYPOTHESIS: Removing the two 8-element output biases will reduce parameters from 1,644 to 1,628 while retaining at least 99% accuracy, because both biases enter residual streams immediately before downstream normalization and the proven-critical `d_ff=12` capacity remains intact.

INTENDED_EDIT: Disable biases in the attention output projection and MLP output projection.

EVIDENCE: Reducing `d_ff` from 12 to 8 cut accuracy to 72.25%, showing that nonlinear width should be preserved; the original `d_ff=12` model achieved 99.96%, leaving margin for a smaller, targeted reduction in comparatively redundant output biases.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
>>>>>>> REPLACE