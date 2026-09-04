MECHANISM: Batch-adaptive normalization for averaged weights

HYPOTHESIS: Using evaluation-batch statistics will exceed 9,285 correct predictions by eliminating the mismatch between tail-averaged parameters and terminal BatchNorm running statistics without adding training-time averaging work.

INTENDED_EDIT: Disable running-stat tracking for every BatchNorm2d layer so training remains unchanged while evaluation computes normalization statistics from the full concatenated ten-view batch.

EVIDENCE: Dense tail averaging achieved the best verified 9,285 correct, while explicitly averaging BatchNorm buffers timed out; batch-adaptive normalization targets the same unresolved parameter/statistics mismatch with no additional optimizer-step copies.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )

        # Tail-averaged weights should be normalized by statistics computed
        # from those weights rather than by statistics from the terminal model.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None
>>>>>>> REPLACE