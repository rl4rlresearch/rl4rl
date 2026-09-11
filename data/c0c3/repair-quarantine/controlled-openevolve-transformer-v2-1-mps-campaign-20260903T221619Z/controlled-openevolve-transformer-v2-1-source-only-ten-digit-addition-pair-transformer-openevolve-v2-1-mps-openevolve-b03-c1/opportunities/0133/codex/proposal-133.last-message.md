MECHANISM: Fixed lexical-radius scale gauge

HYPOTHESIS: Freezing one token radius at its initialized value will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy, because a common rescaling of all token radii can be absorbed exactly by the lexical projection and latent bias, while this edit preserves the successful initialization without the coordinate rotation used by the failed phase gauge.

INTENDED_EDIT: Store the first initialized token radius as a fixed buffer and learn only the remaining radii, reconstructing the full radius vector during every forward pass.

EVIDENCE: The 606-parameter model achieved 99.89% accuracy. The 605-parameter phase gauge fell to 44.29% after rotating lexical coordinates, motivating a scale-gauge fixing that leaves every initialized model value and all relative token radii unchanged.

<<<<<<< SEARCH
        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
=======
        self.token_phase = nn.Parameter(token_phase.clone())
        # A common scale on every radius is exactly absorbed by the inverse
        # scale on token_proj and ln_f_token_bias. Fix one nonzero initialized
        # radius to select a gauge without changing the initialized function.
        self.register_buffer(
            "token_radius_anchor", token_radius[:1].clone()
        )
        self.token_radius = nn.Parameter(token_radius[1:].clone())
        self.token_proj = nn.Parameter(token_proj.clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_code = self.token_radius.unsqueeze(1) * torch.stack(
=======
        token_radius = torch.cat(
            (self.token_radius_anchor, self.token_radius)
        )
        token_code = token_radius.unsqueeze(1) * torch.stack(
>>>>>>> REPLACE