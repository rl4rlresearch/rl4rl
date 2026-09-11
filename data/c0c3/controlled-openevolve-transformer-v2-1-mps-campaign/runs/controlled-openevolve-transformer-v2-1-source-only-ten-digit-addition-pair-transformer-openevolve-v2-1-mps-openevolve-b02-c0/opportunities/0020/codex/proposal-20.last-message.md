MECHANISM: Pre-MLP LayerNorm-to-linear bias gauge anchoring

HYPOTHESIS: Fixing one `ln2` bias coordinate at zero will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because the omitted offset can be absorbed exactly into the following `fc1` bias.

INTENDED_EDIT: Replace the standard pre-MLP LayerNorm with the existing initialization-preserving `AnchoredLayerNorm`, which learns seven bias coordinates and appends one fixed zero.

EVIDENCE: A one-coordinate LayerNorm bias anchor previously achieved 99.95% accuracy, and the current design also achieves 99.95%; applying the same successful reduction before `fc1` preserves initialization and removes an exact LayerNorm-linear parameter redundancy.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = AnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE