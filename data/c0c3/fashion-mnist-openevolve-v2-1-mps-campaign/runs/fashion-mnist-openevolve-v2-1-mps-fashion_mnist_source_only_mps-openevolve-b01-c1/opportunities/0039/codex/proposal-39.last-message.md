MECHANISM: Capacity-completing classifier widening

HYPOTHESIS: Widening the proven flattened classifier from 80 to 88 hidden units will exceed 9,239 correct predictions by using nearly all remaining parameter capacity without adding costly feature-map operations or disrupting the successful backbone and cosine supervision curriculum.

INTENDED_EDIT: Increase the classifier hidden width from 80 to 88, raising learned parameters from 230,442 to 249,362 while leaving training compute, augmentation, loss, and optimization essentially unchanged.

EVIDENCE: The added spatial residual block regressed from 9,239 to 9,188 correct, and a deeper backbone timed out; allocating the remaining capacity to the existing successful head tests representational capacity without repeating either failure mode.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.LayerNorm(88),
            nn.Linear(88, 10),
        )
>>>>>>> REPLACE