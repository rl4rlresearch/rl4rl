MECHANISM: Stacked parameter-efficient residual spatial refinement

HYPOTHESIS: A second depthwise-separable residual block at 7×7 resolution will exceed 9,250 correct predictions by extending the late spatial refinement that improved the previous design, while sacrificing only one dense-head unit and remaining below 250,000 parameters.

INTENDED_EDIT: Stack two SpatialRefinement blocks and reduce the location-sensitive classifier width from 56 to 55, yielding 248,135 learned parameters.

EVIDENCE: Adding the first depthwise-separable residual block improved validation correctness from 9,247 to 9,250 while preserving the spatial dense head; this directly motivates testing another residual refinement stage with minimal loss of head capacity.

<<<<<<< SEARCH
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
        )
=======
            nn.MaxPool2d(2),
            SpatialRefinement(64),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 55),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(55, 10),
        )
>>>>>>> REPLACE