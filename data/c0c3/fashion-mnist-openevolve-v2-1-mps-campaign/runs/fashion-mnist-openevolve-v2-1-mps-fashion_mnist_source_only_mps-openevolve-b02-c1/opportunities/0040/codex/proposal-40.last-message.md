MECHANISM: Low-rank second-order feature co-occurrence branch

HYPOTHESIS: Adding a learned covariance-based scoring branch will exceed 9,286 correct predictions by distinguishing classes with similar first-order shapes through spatial feature co-occurrences, while preserving the proven convolutional representation and tail averaging.

INTENDED_EDIT: Add a 32-channel projection that computes centered channel covariance over the final 3×3 feature map and blends its class logits with the existing flattening classifier through a learned scale.

EVIDENCE: Residual refinements and alternative pooling reduced accuracy, indicating that changing the convolutional backbone or spatial bottleneck is risky; the current design still assumes class predictions need only first-order flattened features, so a computation-light second-order branch tests a genuinely different and complementary representation within the parameter and runtime limits.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
        self.quadratic_features = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.quadratic_classifier = nn.Linear(32 * 32, 10)
        self.quadratic_scale = nn.Parameter(torch.tensor(0.25))

    def _image_logits(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        logits = self.classifier(features)

        quadratic = self.quadratic_features(features).flatten(2)
        quadratic = quadratic - quadratic.mean(dim=2, keepdim=True)
        covariance = torch.bmm(quadratic, quadratic.transpose(1, 2))
        covariance = covariance / quadratic.shape[2]
        quadratic_logits = self.quadratic_classifier(covariance.flatten(1))
        return logits + self.quadratic_scale * quadratic_logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._image_logits(images)
        if self.training:
            return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
=======
        for view in views[1:]:
            view_logits = self._image_logits(view)
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self._image_logits(view.flip(-1))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
>>>>>>> REPLACE