MECHANISM: Affine-free attention pre-normalization

HYPOTHESIS: Removing only `ln1`’s seven learned scales while retaining the full QKV projections will produce a 1,584-parameter model with at least 99% accuracy.

INTENDED_EDIT: Replace the anchored affine `ln1` with parameter-free LayerNorm, leaving QKV weights unconstrained so they can absorb the removed feature scales.

EVIDENCE: The 1,591-parameter design reached 99.96% after the analogous `ln2` scale removal, while the 1,560-parameter `ln1` experiment also gauged all 24 QKV rows and fell to 73.05%; isolating `ln1` scale removal tests which part caused that failure.

<<<<<<< SEARCH
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        # fc1 can absorb LayerNorm's feature scales. Removing the affine scale
        # also makes every fc1 common-row coefficient an exact null direction.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
=======
        # QKV columns can absorb LayerNorm's feature scales; retain the full
        # projections to isolate this reduction from the failed QKV row gauges.
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        # fc1 can absorb LayerNorm's feature scales. Removing the affine scale
        # also makes every fc1 common-row coefficient an exact null direction.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE