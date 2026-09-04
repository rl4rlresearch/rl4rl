MECHANISM: Nine-entry positional-mixer constraint

HYPOTHESIS: Fixing a ninth scalar in the dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,147 to 1,146 learned parameters.

INTENDED_EDIT: Learn 55 of the 64 positional-mixer weights, fixing the final nine entries at zero while preserving full-matrix initialization RNG consumption and the qualified multi-query attention backbone.

EVIDENCE: The current 1,147-parameter multi-query design achieved 100% accuracy, and every incremental positional-mixer constraint from one through eight fixed scalars also achieved 100%; testing one additional scalar is the smallest evidence-backed reduction.

<<<<<<< SEARCH
        # Generic Fourier coordinates are synthesized by the qualified dense
        # map with its final output row fixed at zero.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 8)
        )
        self.pos_mix.fixed_weight_trim = (0, 8)
=======
        # Generic Fourier coordinates are synthesized by the qualified dense
        # map with its final output row and one preceding scalar fixed at zero.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 9)
        )
        self.pos_mix.fixed_weight_trim = (0, 9)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 8)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 9)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
>>>>>>> REPLACE