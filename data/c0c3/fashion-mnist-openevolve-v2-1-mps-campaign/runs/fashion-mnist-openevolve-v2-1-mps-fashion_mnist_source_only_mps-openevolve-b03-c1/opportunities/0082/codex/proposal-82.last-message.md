MECHANISM: Fixed-energy adaptive detail filters

HYPOTHESIS: Constraining each zero-DC detail kernel to its initial L2 norm will exceed 9,166 correct predictions by removing scale drift while retaining task-specific filter-shape adaptation.

INTENDED_EDIT: Store the three initial kernel norms and renormalize each mean-centered learned kernel before convolution, without changing initialization, parameter count, or training cost materially.

EVIDENCE: Learning zero-DC detail filters improved correctness from 9,162 to 9,166, while adding gradient magnitude regressed to 9,114; preserving the successful adaptive representation while eliminating its scale ambiguity is the most direct refinement.

<<<<<<< SEARCH
        )
        self.features = nn.Sequential(
=======
        )
        self.register_buffer(
            "detail_kernel_norms",
            torch.tensor(
                [math.sqrt(0.1875), math.sqrt(0.1875), math.sqrt(1.25)],
                dtype=torch.float32,
            ).view(3, 1, 1, 1),
        )
        self.features = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        detail_kernels = self.detail_kernels - self.detail_kernels.mean(
            dim=(2, 3), keepdim=True
        )
        details = F.conv2d(padded, detail_kernels)
=======
        detail_kernels = self.detail_kernels - self.detail_kernels.mean(
            dim=(2, 3), keepdim=True
        )
        detail_kernels = detail_kernels * (
            self.detail_kernel_norms
            / detail_kernels.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        )
        details = F.conv2d(padded, detail_kernels)
>>>>>>> REPLACE