MECHANISM: Broader grouped-convolution channel mixing

HYPOTHESIS: Expanding each refinement group from two to four channels while slightly narrowing the dense layer will raise validation_correct above 9,281 by improving spatial feature interaction within the parameter ceiling.

INTENDED_EDIT: Change the refinement convolution from 64 to 32 groups, reduce the classifier width from 128 to 123, and use the best-verified evaluation scale; the resulting model has 249,011 learned parameters.

EVIDENCE: The parameter-neutral residual refinement reduced accuracy to 9,269, while smoothing and calibration searches plateaued at 9,281; reallocating capacity from the large dense layer into the existing spatial refinement tests stronger channel mixing without the unsuccessful skip connection.

<<<<<<< SEARCH
        self.conv = nn.Conv2d(
            128, 128, kernel_size=5, padding=2, groups=64, bias=False
        )
=======
        self.conv = nn.Conv2d(
            128, 128, kernel_size=5, padding=2, groups=32, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 10),
=======
            nn.Linear(128 * 3 * 3, 123),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(123, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            logits = 1.19 * logits
=======
        if not self.training:
            logits = 1.20485 * logits
>>>>>>> REPLACE