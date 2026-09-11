MECHANISM: Identity-initialized expanded-support adaptive detail filters

HYPOTHESIS: Expanding the successful zero-DC learned detail kernels from 3×3 to 5×5 will exceed 9,166 correct predictions by learning longer-range edge and texture structure while preserving the verified model exactly at initialization.

INTENDED_EDIT: Zero-pad each learned derivative kernel to 5×5 and increase the replicate padding accordingly, adding 48 learned parameters for an estimated total of 249,837.

EVIDENCE: Learning the original zero-DC 3×3 filters improved correctness from 9,162 to 9,166, whereas fixing their energy regressed to 9,153; this motivates adding unconstrained filter-shape capacity without changing the initial computation.

<<<<<<< SEARCH
        self.detail_kernels = nn.Parameter(
            torch.tensor(
                [
                    [
                        [
                            [-0.125, 0.0, 0.125],
                            [-0.250, 0.0, 0.250],
                            [-0.125, 0.0, 0.125],
                        ]
                    ],
                    [
                        [
                            [-0.125, -0.250, -0.125],
                            [0.0, 0.0, 0.0],
                            [0.125, 0.250, 0.125],
                        ]
                    ],
                    [
                        [
                            [0.0, 0.250, 0.0],
                            [0.250, -1.0, 0.250],
                            [0.0, 0.250, 0.0],
                        ]
                    ],
                ],
                dtype=torch.float32,
            )
        )
=======
        self.detail_kernels = nn.Parameter(
            torch.tensor(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, -0.125, 0.0, 0.125, 0.0],
                            [0.0, -0.250, 0.0, 0.250, 0.0],
                            [0.0, -0.125, 0.0, 0.125, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ],
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, -0.125, -0.250, -0.125, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.125, 0.250, 0.125, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ],
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.250, 0.0, 0.0],
                            [0.0, 0.250, -1.0, 0.250, 0.0],
                            [0.0, 0.0, 0.250, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ],
                ],
                dtype=torch.float32,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        detail_kernels = self.detail_kernels - self.detail_kernels.mean(
=======
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        detail_kernels = self.detail_kernels - self.detail_kernels.mean(
>>>>>>> REPLACE