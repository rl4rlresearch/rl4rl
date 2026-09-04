MECHANISM: Deterministic multi-scale local-contrast basis

HYPOTHESIS: Adding a 5×5 contrast channel will exceed 9,283 correct predictions by exposing medium-scale garment structure that the initial 3×3 convolution cannot directly observe, while preserving the verified backbone and optimization settings.

INTENDED_EDIT: Expand the fixed image basis from five to six channels with 5×5 local contrast, adding only 216 learned parameters.

EVIDENCE: The 245,150-parameter baseline reached 9,283 correct, while learned non-local attention fell to 9,265 and covariance augmentation to 9,239; this motivates a low-capacity deterministic feature that supplies broader context without replacing or compressing proven components.

<<<<<<< SEARCH
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(6, 24, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
        local_mean = F.avg_pool2d(padded, kernel_size=3, stride=1)
        local_contrast = images - local_mean

        gradient_x = 0.125 * (
=======
        local_mean = F.avg_pool2d(padded, kernel_size=3, stride=1)
        local_contrast = images - local_mean
        broad_mean = F.avg_pool2d(
            F.pad(images, (2, 2, 2, 2), mode="replicate"),
            kernel_size=5,
            stride=1,
        )
        broad_contrast = images - broad_mean

        gradient_x = 0.125 * (
>>>>>>> REPLACE

<<<<<<< SEARCH
                images,
                local_contrast,
                gradient_x,
=======
                images,
                local_contrast,
                broad_contrast,
                gradient_x,
>>>>>>> REPLACE