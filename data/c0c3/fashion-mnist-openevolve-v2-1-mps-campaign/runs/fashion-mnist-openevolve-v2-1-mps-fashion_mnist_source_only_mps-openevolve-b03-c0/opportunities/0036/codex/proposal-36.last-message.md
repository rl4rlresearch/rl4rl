MECHANISM: Normalized dense-head optimization

HYPOTHESIS: Adding BatchNorm to the dense projection will exceed 9,319 correct predictions by improving short-horizon optimization while retaining nearly all classifier capacity.

INTENDED_EDIT: Reduce the hidden width from 44 to 43 and normalize its activations with BatchNorm1d before SiLU, remaining below the parameter ceiling.

EVIDENCE: Spatial refinements improved cross-entropy but lost accuracy when classifier width fell to 32; this instead preserves the dominant flattened head and adds inexpensive optimization support not tested by prior scheduling, smoothing, or endpoint-averaging changes.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 44),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(44, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 43),
            nn.BatchNorm1d(43),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(43, 10),
        )
>>>>>>> REPLACE