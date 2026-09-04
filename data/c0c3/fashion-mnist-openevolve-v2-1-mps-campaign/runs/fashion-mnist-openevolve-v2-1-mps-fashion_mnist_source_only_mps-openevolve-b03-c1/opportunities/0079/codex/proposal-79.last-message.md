MECHANISM: Zero-DC adaptive detail filter bank

HYPOTHESIS: Learning the existing three derivative kernels while constraining each to remain zero-mean will exceed 9,162 correct predictions by adapting edge and texture extraction to the dataset without adding the redundant gradient-magnitude channel that regressed to 9,114.

INTENDED_EDIT: Convert the 27 fixed detail-kernel coefficients into learned parameters and remove each kernel’s spatial mean before convolution, preserving the original computation at initialization and raising the parameter count to 249,789.

EVIDENCE: The fixed derivative representation supports the 9,162-correct design, whereas adding a handcrafted gradient-magnitude channel reduced correctness to 9,114; adapting the successful filters is a lightweight alternative to expanding the representation.

<<<<<<< SEARCH
        self.register_buffer(
            "detail_kernels",
            torch.tensor(
=======
        self.detail_kernels = nn.Parameter(
            torch.tensor(
>>>>>>> REPLACE

<<<<<<< SEARCH
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.features = nn.Sequential(
=======
                dtype=torch.float32,
            )
        )
        self.features = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        details = F.conv2d(padded, self.detail_kernels)
=======
        detail_kernels = self.detail_kernels - self.detail_kernels.mean(
            dim=(2, 3), keepdim=True
        )
        details = F.conv2d(padded, detail_kernels)
>>>>>>> REPLACE