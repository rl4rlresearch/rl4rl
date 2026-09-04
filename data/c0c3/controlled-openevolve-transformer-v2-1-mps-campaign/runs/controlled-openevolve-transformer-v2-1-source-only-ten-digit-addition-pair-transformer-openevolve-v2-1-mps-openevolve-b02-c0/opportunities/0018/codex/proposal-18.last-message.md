MECHANISM: Residual-stream common-mode bias anchoring

HYPOTHESIS: Fixing one coordinate of the attention output-projection bias at zero will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because the omitted common-mode offset is removed by both the downstream pre-MLP LayerNorm and final LayerNorm.

INTENDED_EDIT: Reuse the initialization-preserving `OutputAnchoredLinear` for the attention output projection, learning seven bias coordinates and appending one fixed-zero coordinate.

EVIDENCE: Anchoring one coordinate of the final MLP output bias achieved 99.95% accuracy at 1,635 parameters, demonstrating that removing a common-mode residual-stream bias degree of freedom before final normalization preserves performance; the attention projection has the same gauge because its common offset survives only in the residual stream while normalized downstream computations remain unchanged.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = OutputAnchoredLinear(d_model, d_model)
>>>>>>> REPLACE