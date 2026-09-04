MECHANISM: Fixed orthogonal recurrent-coordinate folding

HYPOTHESIS: Folding all 98 recurrent summary coordinates into a 96-dimensional readout will retain at least 85% accuracy while reducing classifier inference by 6,520 MACs versus the verified 97-coordinate design.

INTENDED_EDIT: Reduce the classifier input from 97 to 96 and fold the final two summary coordinates into the first two with variance-preserving fixed additions, preserving a direct classification path from every recurrent coordinate.

EVIDENCE: The cropped 96-coordinate readout narrowly failed at 84.66% despite achieving lower cross-entropy than the passing 97-coordinate model; retaining information and direct classifier gradients from the two previously discarded coordinates specifically targets that failure while keeping the lower 96-input MAC count.

<<<<<<< SEARCH
        self.classifier = nn.Linear(97, 8)
=======
        self.classifier = nn.Linear(96, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        averaged = summary / count.clamp_min(1.0)
        return self.classifier(averaged[:, :97])
=======
        averaged = summary / count.clamp_min(1.0)
        folded = torch.cat(
            (
                (averaged[:, :2] + averaged[:, 96:98]) * (2.0**-0.5),
                averaged[:, 2:96],
            ),
            dim=1,
        )
        return self.classifier(folded)
>>>>>>> REPLACE