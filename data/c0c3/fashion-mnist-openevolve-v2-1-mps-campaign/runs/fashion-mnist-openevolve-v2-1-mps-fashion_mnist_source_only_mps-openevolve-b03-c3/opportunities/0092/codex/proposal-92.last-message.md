MECHANISM: Identity-initialized residual refinement

HYPOTHESIS: Zero-initializing the existing refinement branch’s final BatchNorm scale while restoring the best verified TTA weights will exceed 9,268 correct predictions by preserving the successful topology and preventing random residual perturbations during early optimization.

INTENDED_EDIT: Restore Reference Design 1’s center and radius-1 diagonal TTA weights, and initialize the refinement residual as an exact identity without changing parameters or compute.

EVIDENCE: The 40-local/24-context model with 0.36/0.045 TTA achieved the best 9,268 correct; replacement heads and added channel gating regressed, motivating an optimization-focused change that retains the verified architecture.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
        nn.init.zeros_(self.refine[-1].weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = 0.38 * self._flip_average(images)
=======
        logits = 0.36 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.04
=======
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
>>>>>>> REPLACE