MECHANISM: Rank-aligned final LayerNorm bias

HYPOTHESIS: Reparameterizing the final eight-dimensional LayerNorm bias as a six-dimensional token-feature bias will reduce parameters from 1,223 to 1,221 while retaining at least 99% accuracy, because logits observe the final hidden state only through the full-rank six-dimensional lexical projection.

INTENDED_EDIT: Remove the final LayerNorm’s redundant eight-coordinate bias and add a learned six-coordinate bias immediately after projection into token-feature space.

EVIDENCE: The gauge-fixed rank-six lexical bottleneck achieved 99.96% accuracy at 1,223 parameters, and the successful direct attention-output-bias reparameterization showed that moving a bias into its observable projected space can preserve accuracy; this targets only the lexical bottleneck’s exact two-dimensional output nullspace.

<<<<<<< SEARCH
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Initialize a dense tied lexical table as before, then compress it
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # The final state reaches the logits only through a rank-six lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining two null directions in the LayerNorm bias.
        token_rank = 6
        self.ln_f_token_bias = nn.Parameter(
            self.ln_f.bias.new_zeros(token_rank)
        )
        self.ln_f.bias = None

        # Initialize a dense tied lexical table as before, then compress it
>>>>>>> REPLACE

<<<<<<< SEARCH
        # six feature columns as an identity pivot and learning only the two
        # remaining columns.
        token_rank = 6
        with torch.no_grad():
=======
        # six feature columns as an identity pivot and learning only the two
        # remaining columns.
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_features = x @ token_proj.T
        logits = F.linear(token_features, self.token_code)
=======
        token_features = x @ token_proj.T + self.ln_f_token_bias
        logits = F.linear(token_features, self.token_code)
>>>>>>> REPLACE