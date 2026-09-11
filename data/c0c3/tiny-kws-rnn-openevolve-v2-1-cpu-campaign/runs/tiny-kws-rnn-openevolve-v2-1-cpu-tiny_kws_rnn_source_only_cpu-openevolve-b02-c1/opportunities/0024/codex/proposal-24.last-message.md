MECHANISM: Orthonormal softmax gauge fixing

HYPOTHESIS: An orthonormal seven-coordinate parameterization of the eight centered logits will retain at least 85% accuracy while matching the prior seven-logit MAC and parameter reductions.

INTENDED_EDIT: Replace the eight-output classifier with seven learned contrast coordinates and expand them through a fixed orthonormal basis into eight zero-sum logits.

EVIDENCE: The fixed-zero seven-logit model reached 84.42% despite preserving softmax expressivity, suggesting its asymmetric parameterization hurt optimization; balanced orthonormal contrasts preserve the same full function class with better-conditioned, nonconstant logits for every class.

<<<<<<< SEARCH
        self.classifier = nn.Linear(119, 8)
=======
        self.classifier = nn.Linear(119, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden_a, _hidden_b, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        _hidden_a, _hidden_b, summary, count = state
        coordinates = self.classifier(summary / count.clamp_min(1.0))
        coordinate_sum = coordinates.sum(dim=1, keepdim=True)
        scale = 1.0 / math.sqrt(8.0)
        correction = (scale * scale) / (1.0 - scale)
        return torch.cat(
            (
                coordinates - correction * coordinate_sum,
                scale * coordinate_sum,
            ),
            dim=1,
        )
>>>>>>> REPLACE