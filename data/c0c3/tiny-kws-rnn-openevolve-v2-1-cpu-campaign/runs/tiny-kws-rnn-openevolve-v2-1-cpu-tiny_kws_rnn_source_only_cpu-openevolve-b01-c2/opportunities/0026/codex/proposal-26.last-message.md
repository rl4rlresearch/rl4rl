MECHANISM: Reference-class softmax with endpoint-preserving frame pruning

HYPOTHESIS: A seven-logit reference-class head is expressively equivalent under softmax and will preserve at least 85% accuracy on the qualified 25-step schedule while reducing total dense inference MACs from 294,573,600 to approximately 294,426,900.

INTENDED_EDIT: Adopt frames 3–26 plus frame 28, and replace the redundant eight-output affine classifier with seven learned relative logits and one zero reference logit.

EVIDENCE: The 25-step endpoint-preserving schedule achieved 85.15% accuracy at 294,573,600 MACs. Softmax is invariant to a shared logit offset, so eight-class probabilities require only seven learned logit contrasts.

<<<<<<< SEARCH
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.classifier = nn.Linear(3 * self.hidden_size, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(features)
=======
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
>>>>>>> REPLACE