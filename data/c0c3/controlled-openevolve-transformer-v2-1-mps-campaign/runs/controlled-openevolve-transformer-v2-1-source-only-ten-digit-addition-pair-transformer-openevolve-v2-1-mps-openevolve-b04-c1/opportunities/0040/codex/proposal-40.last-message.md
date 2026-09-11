MECHANISM: Gated recurrent transformer refinement

HYPOTHESIS: Reusing the learned causal transformer block for a gated second refinement step will maintain at least 99% accuracy with 1587 parameters, because recurrent depth can recover the nonlinear and cross-token capacity lost by narrowing the MLP from 12 to 11 channels.

INTENDED_EDIT: Narrow the MLP by one channel and add a single learned, zero-initialized gate that lets the same attention-and-MLP block iteratively refine its first-pass state without duplicating weights.

EVIDENCE: The seven-channel one-pass SwiGLU fell to 53.57%, indicating that simply compressing the token-local mixer removes load-bearing capacity; meanwhile repeated exact gauge reductions became optimization-fragile at 1602 parameters. This challenges the shared single-pass-width assumption by trading one MLP channel for parameter-shared computational depth while preserving the successful 1603-parameter model at initialization up to the modest width change.

<<<<<<< SEARCH
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.refinement_gate = nn.Parameter(torch.zeros(()))
        self.ln_f = nn.LayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
=======
        for blk in self.blocks:
            x = blk(x)
            refined = blk(x)
            x = x + torch.tanh(self.refinement_gate) * (refined - x)

        x = self.ln_f(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE