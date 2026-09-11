MECHANISM: Mid-level statistical feature bypass

HYPOTHESIS: Feeding global average and maximum statistics from the 64-channel intermediate feature map directly to the classifier will exceed 9,311 correct predictions by preserving texture and part evidence otherwise discarded by the final convolution and pooling stage.

INTENDED_EDIT: Split the encoder into stages, concatenate the deepest spatial representation with intermediate channel-wise mean and maximum descriptors, and resize the hidden layer to remain below the parameter ceiling.

EVIDENCE: Parameter-free residual skips fell to 9,295 correct while still classifying solely from the deepest feature map, suggesting that modifying feature refinement alone is insufficient. This patch challenges the shared assumption that all useful evidence must survive the final 96-channel spatial compression by creating a direct multi-scale path to class prediction.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )
=======
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 3 * 3 + 2 * 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.25
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.25
            ensemble = ensemble + flipped_logits
        return ensemble / (12.5 * 0.9350)
=======
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        shallow = self.stage1(images)
        middle = self.stage2(shallow)
        deep = self.stage3(middle)
        representation = torch.cat(
            (
                deep.flatten(1),
                middle.mean(dim=(2, 3)),
                middle.amax(dim=(2, 3)),
            ),
            dim=1,
        )
        return self.classifier(representation)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.25
        for view in views[1:]:
            view_logits = self._classify(view)
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self._classify(view.flip(-1))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.25
            ensemble = ensemble + flipped_logits
        return ensemble / (12.5 * 0.9350)
>>>>>>> REPLACE