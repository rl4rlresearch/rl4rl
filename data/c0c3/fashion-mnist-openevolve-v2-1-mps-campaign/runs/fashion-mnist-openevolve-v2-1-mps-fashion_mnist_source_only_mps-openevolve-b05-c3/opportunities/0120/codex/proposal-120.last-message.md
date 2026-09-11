MECHANISM: Baseline-preserving channel-spatial residual routing

HYPOTHESIS: Image-conditioned channel and spatial gating of residual refinements will exceed 9,206 correct predictions by suppressing irrelevant texture and emphasizing class-defining shape regions, while retaining the qualified split EMA and calibrated TTA.

INTENDED_EDIT: Replace the static convolutional stack with residual stages whose refinement paths receive learned channel-and-spatial attention, initialized to reproduce ordinary residual addition; retain nearly all classifier capacity with a 47-unit head and use the best verified EMA and TTA calibration.

EVIDENCE: Residual refinement improved correctness from 9,168 to 9,206, but all qualified designs assume every image uses the same refinement computation. The late global-moment branch reached only 9,201, suggesting global context is more useful when it modulates features inside the spatial hierarchy; the zero-initialized lightweight gates test that alternative without the runtime cost of the failed adaptive-pooling design.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 96))
        for index, (in_channels, out_channels) in enumerate(channels):
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            probability_sum.add_(F.softmax(self._forward_once(view), dim=1))
            probability_sum.add_(
                F.softmax(self._forward_once(view.flip(-1)), dim=1)
            )

        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.channel_gate = nn.Conv1d(
            1,
            1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.spatial_gate = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.zeros_(self.spatial_gate.weight)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))

        channel_context = refinement.mean(dim=(2, 3), keepdim=False)
        channel_scale = self.channel_gate(
            channel_context.unsqueeze(1)
        ).squeeze(1)
        channel_scale = 2.0 * torch.sigmoid(
            channel_scale[:, :, None, None]
        )

        spatial_context = torch.cat(
            (
                refinement.mean(dim=1, keepdim=True),
                refinement.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_scale = 2.0 * torch.sigmoid(
            self.spatial_gate(spatial_context)
        )
        refinement = refinement * channel_scale * spatial_scale
        return self.pool(F.silu(features + refinement, inplace=True))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualStage(1, 32),
            ResidualStage(32, 64),
            ResidualStage(64, 96),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 47),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(47, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
=======
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    optimizer._ema_tensors = [
        tensor
        for tensor in parameters + buffers
        if tensor.is_floating_point()
    ]
    buffer_ids = {id(tensor) for tensor in buffers}
    optimizer._ema_is_buffer = [
        id(tensor) in buffer_ids for tensor in optimizer._ema_tensors
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
=======
        else:
            terminal_phase = step * 32 >= total_steps * 19
            for average, tensor, is_buffer in zip(
                optimizer._ema_values,
                optimizer._ema_tensors,
                optimizer._ema_is_buffer,
            ):
                if terminal_phase:
                    ema_rate = 0.0325 if is_buffer else 0.015
                else:
                    ema_rate = 0.005
                average.lerp_(tensor, ema_rate)
>>>>>>> REPLACE