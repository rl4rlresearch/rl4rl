MECHANISM: Higher-update residual refinement

HYPOTHESIS: Halving the unique-image batch size while adopting the proven two-convolution residual block will exceed Reference Design 2’s 9,310 correct predictions by providing roughly twice as many optimizer updates over the same 100,000 examples.

INTENDED_EDIT: Restore the best 249,393-parameter residual architecture and non-EMA schedule, then reduce batch size from 128 to 64.

EVIDENCE: Reference Design 2 achieved the best result at 9,310 correct; channel attention reduced this to 9,296 and EMA reduced the earlier architecture to 9,268, motivating retention of the proven model while testing increased update frequency.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
    optimizer._ema_parameters = []
    return optimizer
=======
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    ema_parameters = optimizer._ema_parameters
    with torch.no_grad():
        if not ema_parameters:
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            ema_parameters = optimizer._ema_parameters
        else:
            for ema_parameter, parameter in zip(ema_parameters, parameters):
                ema_parameter.mul_(0.99).add_(parameter, alpha=0.01)

        if step + 1 == total_steps:
            for parameter, ema_parameter in zip(parameters, ema_parameters):
                parameter.copy_(ema_parameter)

    peak_lr = 3.0e-3
=======
    peak_lr = 3.0e-3
>>>>>>> REPLACE