MECHANISM: Additive rank-four class-specific spatial-part evidence head

HYPOTHESIS: A direct class-specific route that factorizes channel evidence and learned spatial templates will exceed 9,267 correct predictions by preserving garment layout while avoiding exclusive reliance on the shared 64-dimensional flattened bottleneck.

INTENDED_EDIT: Retain the existing classifier and add four learned part maps and spatial templates per class, contributing additive logits through a bounded learned scale; total parameters become 237,715.

EVIDENCE: Position-free pooling regressed to 9,253/9,249, showing layout is load-bearing, while widening the flattened head regressed to 9,221 and attention timed out. This tests structured, class-specific spatial evidence without discarding the successful dense head or adding expensive input-dependent attention.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
        # A complementary low-rank classifier gives every class a direct route
        # from channel evidence at learned garment-part locations.  Four
        # channel/spatial factors per class preserve layout without another
        # large flattened projection.
        self.part_maps = nn.Conv2d(56, 10 * 4, kernel_size=1)
        self.part_locations = nn.Parameter(torch.zeros(10, 4, 7, 7))
        self.part_mixture = nn.Parameter(torch.zeros(10, 4))
        self.part_scale = nn.Parameter(torch.tensor(0.0))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        dense_logits = self.classifier(features)

        part_maps = self.part_maps(features).reshape(
            features.shape[0], 10, 4, 7, 7
        )
        location_weights = F.softmax(
            self.part_locations.flatten(-2), dim=-1
        ).reshape(1, 10, 4, 7, 7)
        part_evidence = (part_maps * location_weights).sum(dim=(-1, -2))
        mixture_weights = F.softmax(
            self.part_mixture, dim=-1
        ).unsqueeze(0)
        spatial_logits = (part_evidence * mixture_weights).sum(dim=-1)
        return dense_logits + self.part_scale.sigmoid() * spatial_logits
>>>>>>> REPLACE