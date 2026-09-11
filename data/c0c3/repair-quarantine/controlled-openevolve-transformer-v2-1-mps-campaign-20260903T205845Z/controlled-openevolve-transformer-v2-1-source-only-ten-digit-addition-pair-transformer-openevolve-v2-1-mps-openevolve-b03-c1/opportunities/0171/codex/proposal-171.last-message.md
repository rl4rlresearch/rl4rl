MECHANISM: Lexical global-scale gauge fixing

HYPOTHESIS: Fixing one token-radius coordinate will reduce the verified 606-parameter model to 605 parameters while retaining at least 99% accuracy, because a reciprocal rescaling of all token radii, the shared lexical projection, and the final lexical bias makes this coordinate functionally redundant while preserving the initialized model exactly.

INTENDED_EDIT: Store the first initialized token radius as a fixed buffer and learn only the remaining radii, reconstructing the complete radius vector during the forward pass.

EVIDENCE: The 606-parameter design achieved 0.9989 accuracy, while the failed 604-parameter lexical experiment changed the lexical representation itself; this patch instead removes only an exact continuous scale redundancy without changing its initialization or representable learned functions.

<<<<<<< SEARCH
        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
=======
        self.token_phase = nn.Parameter(token_phase.clone())
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