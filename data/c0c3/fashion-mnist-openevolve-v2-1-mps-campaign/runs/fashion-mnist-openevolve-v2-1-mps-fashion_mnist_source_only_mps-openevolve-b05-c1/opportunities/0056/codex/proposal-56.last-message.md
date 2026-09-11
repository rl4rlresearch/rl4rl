MECHANISM: Normalized spatial first-moment residual head

HYPOTHESIS: Adding class-specific horizontal and vertical first moments will exceed 9,334 correct predictions by preserving coarse positional evidence omitted by the translation-invariant statistics head.

INTENDED_EDIT: Add a zero-initialized residual classifier over normalized per-channel horizontal and vertical feature moments, increasing learned parameters from 247,418 to 248,964.

EVIDENCE: The global-statistics bypass improved correctness from 9,314 to 9,334, showing that direct class-specific paths around the bottleneck help; prior coarse-spatial proposals were never verified, so positional evidence remains an informative untested complement.

<<<<<<< SEARCH
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
=======
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)

        coordinates = torch.linspace(-1.0, 1.0, 7)
        self.register_buffer(
            "horizontal_coordinates", coordinates.view(1, 1, 1, 7)
        )
        self.register_buffer(
            "vertical_coordinates", coordinates.view(1, 1, 7, 1)
        )
        self.position_norm = nn.BatchNorm1d(64 * 2)
        self.position_head = nn.Linear(64 * 2, 10)
        nn.init.zeros_(self.position_head.weight)
        nn.init.zeros_(self.position_head.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(feature_map) + residual_logits
=======
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        horizontal_moment = (
            feature_map * self.horizontal_coordinates
        ).mean(dim=(2, 3))
        vertical_moment = (
            feature_map * self.vertical_coordinates
        ).mean(dim=(2, 3))
        position = torch.cat(
            (horizontal_moment, vertical_moment), dim=1
        )
        position_logits = self.position_head(
            self.position_norm(position)
        )
        return self.classifier(feature_map) + residual_logits + position_logits
>>>>>>> REPLACE