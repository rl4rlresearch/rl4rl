MECHANISM: LayerNorm positional shift gauge fixing

HYPOTHESIS: Centering each harmonic readout row and reconstructing its eighth coordinate as the negative sum of the other seven will reduce the model from 1,494 to 1,490 parameters while retaining at least 99% accuracy, because per-position shifts along the all-ones feature direction are exactly removed by every downstream LayerNorm.

INTENDED_EDIT: Replace the learned 4-by-8 positional readout with a centered 4-by-7 parameterization and reconstruct the omitted coordinates during the forward pass.

EVIDENCE: Fixed Fourier coordinates with a dense learned readout achieved 99.96% accuracy at 1,494 parameters; this preserves that successful positional representation while removing four functionally unobservable LayerNorm shift directions.

<<<<<<< SEARCH
        self.register_buffer("pos_code", pos_code)
        self.pos_proj = nn.Parameter(self.pos_emb.weight[:4].detach().clone())
        self.pos_emb = None
=======
        self.register_buffer("pos_code", pos_code)
        # A position-dependent shift shared by all model coordinates is removed
        # by every downstream LayerNorm. Center each readout row to fix this
        # exact gauge, then omit its final coordinate.
        pos_proj = self.pos_emb.weight[:4].detach().clone()
        pos_proj = pos_proj - pos_proj.mean(dim=1, keepdim=True)
        self.pos_proj = nn.Parameter(pos_proj[:, :-1].clone())
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        position = F.embedding(pos, self.pos_code) @ self.pos_proj
        x = self.token_emb(idx) + position
=======
        pos_proj = torch.cat(
            (self.pos_proj, -self.pos_proj.sum(dim=1, keepdim=True)), dim=1
        )
        position = F.embedding(pos, self.pos_code) @ pos_proj
        x = self.token_emb(idx) + position
>>>>>>> REPLACE