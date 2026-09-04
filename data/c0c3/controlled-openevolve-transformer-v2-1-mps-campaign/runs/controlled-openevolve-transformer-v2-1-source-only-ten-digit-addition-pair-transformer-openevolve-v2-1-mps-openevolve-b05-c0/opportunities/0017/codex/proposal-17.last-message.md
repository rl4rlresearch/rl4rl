MECHANISM: Quotient-aware global positional-embedding gauge fixing

HYPOTHESIS: Removing the single global common-shift degree of freedom from the positional embedding will reduce the model from 1633 to 1632 parameters while retaining at least 99% accuracy, because this shift is canceled by LayerNorm and the successful virtual full-coordinate AdamW treatment preserves the original embedding’s optimization dynamics.

INTENDED_EDIT: Represent the entire positional embedding with `GaugeFixedEmbedding` and optimize it alongside the tied token embedding and attention output bias using virtual full-coordinate moments and clipping.

EVIDENCE: The analogous global gauge removal from the tied token embedding achieved 99.72% at 1633 parameters. Unlike the failed final-row positional gauge at 69.77%, this applies the exact whole-embedding quotient structure already verified successfully.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
=======
    gauge_parameters = [
        model.token_emb.weight,
        model.pos_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
>>>>>>> REPLACE