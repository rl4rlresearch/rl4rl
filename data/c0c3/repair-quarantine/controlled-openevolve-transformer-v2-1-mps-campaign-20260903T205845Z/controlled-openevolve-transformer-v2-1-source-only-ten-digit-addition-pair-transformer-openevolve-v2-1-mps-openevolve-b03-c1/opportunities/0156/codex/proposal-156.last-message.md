MECHANISM: Gauge-fixed harmonic lexical factorization

HYPOTHESIS: Freezing one token phase and radius at their freshly initialized values will reduce the model from 606 to 604 parameters while retaining at least 99% accuracy, because global phase rotation and radius scaling are exact redundancies absorbed by `token_proj`.

INTENDED_EDIT: Store the first token’s initialized phase and radius as fixed buffers and learn only the remaining token coordinates.

EVIDENCE: The verified 606-parameter design achieved 0.9989 accuracy, and its lexical map depends only on the products of harmonic token codes with `token_proj`, exposing two removable continuous gauges without reducing its function class or changing its initialized function.

<<<<<<< SEARCH
        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
        self.token_emb = None
        self.lm_head = None
=======
        # Global phase rotations and uniform radius rescalings can be absorbed
        # exactly into token_proj. Fix both gauges at their freshly initialized
        # values while preserving the initialized lexical map.
        self.register_buffer(
            "token_anchor_phase", token_phase[:1].clone()
        )
        self.register_buffer(
            "token_anchor_radius", token_radius[:1].clone()
        )
        self.token_phase = nn.Parameter(token_phase[1:].clone())
        self.token_radius = nn.Parameter(token_radius[1:].clone())
        self.token_proj = nn.Parameter(token_proj.clone())
        self.token_emb = None
        self.lm_head = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_code = self.token_radius.unsqueeze(1) * torch.stack(
            (
                torch.sin(self.token_phase),
                torch.cos(self.token_phase),
                torch.sin(2.0 * self.token_phase),
                torch.cos(2.0 * self.token_phase),
            ),
            dim=1,
        ) / math.sqrt(2.0)
=======
        token_phase = torch.cat(
            (self.token_anchor_phase, self.token_phase)
        )
        token_radius = torch.cat(
            (self.token_anchor_radius, self.token_radius)
        )
        token_code = token_radius.unsqueeze(1) * torch.stack(
            (
                torch.sin(token_phase),
                torch.cos(token_phase),
                torch.sin(2.0 * token_phase),
                torch.cos(2.0 * token_phase),
            ),
            dim=1,
        ) / math.sqrt(2.0)
>>>>>>> REPLACE