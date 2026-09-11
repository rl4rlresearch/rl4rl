MECHANISM: Shared reparameterizable MLP/LayerNorm bias

HYPOTHESIS: Tying the second pre-LayerNorm bias to the MLP output bias will reduce parameters from 1628 to 1620 while retaining at least 99% accuracy, because the MLP input bias can absorb any LayerNorm-bias effect while both optimization pathways remain active.

INTENDED_EDIT: Make `ln2.bias` and `mlp.fc2.bias` reference the same learned 8-element parameter.

EVIDENCE: Sharing the redundant attention value/output biases achieved 99.76% at 1628 parameters, whereas deleting the value bias collapsed accuracy; this motivates another pathway-preserving tie instead of removing a redundant bias.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.ln2.bias = self.mlp.fc2.bias
>>>>>>> REPLACE