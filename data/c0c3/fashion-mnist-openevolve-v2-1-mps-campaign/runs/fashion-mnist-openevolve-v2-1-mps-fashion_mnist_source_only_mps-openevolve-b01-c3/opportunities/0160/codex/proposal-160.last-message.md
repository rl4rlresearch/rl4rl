MECHANISM: Reflection-orbit stable/sensitive feature fusion

HYPOTHESIS: Exact reflection-invariant fusion of aligned late feature maps, while retaining absolute orientation-sensitive discrepancies, will exceed the 9,312 correct predictions of soft flip-consistency training.

INTENDED_EDIT: Replace independent flip-logit averaging with a learned classifier over reflection-stable and reflection-sensitive feature statistics; evaluate only the five distinct offset orbits and retain the best validated agreement calibration.

EVIDENCE: Paired flip-consistency training improved correctness from 9,289 to 9,312, showing that reflection instability is load-bearing. The old designs assume augmentation, regularization, and output averaging are sufficient; this patch instead makes reflection-orbit structure part of the image representation without discarding informative asymmetric filter responses.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
        features = self.down1(self.pool1(features))
        features = self.mid_context(features)
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 30),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(30, 10),
        )

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
        features = self.down1(self.pool1(features))
        features = self.mid_context(features)
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        return self.late_refinement(features)

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        paired_features = self._encode(
            torch.cat((images, images.flip(-1)), dim=0)
        )
        original_features, reflected_features = paired_features.chunk(2, dim=0)
        reflected_features = reflected_features.flip(-1)

        stable_features = 0.5 * (
            original_features + reflected_features
        )
        sensitive_features = 0.5 * (
            original_features - reflected_features
        ).abs()
        orbit_features = torch.cat(
            (stable_features, sensitive_features),
            dim=1,
        )
        mean_features = F.adaptive_avg_pool2d(
            orbit_features, 1
        ).flatten(1)
        peak_features = F.adaptive_max_pool2d(
            orbit_features, 1
        ).flatten(1)
        return self.classifier(
            torch.cat((mean_features, peak_features), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = []
        for row_offset, col_offset in (
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
        ):
            view = padded[
                :,
                :,
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            views.append(view)
            views.append(view.flip(-1))

        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
=======
        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = []
        for row_offset, col_offset in (
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
        ):
            views.append(
                padded[
                    :,
                    :,
                    row_offset : row_offset + height,
                    col_offset : col_offset + width,
                ]
            )

        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, images.shape[0], 10
        )
        view_weights = view_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).view(5, 1, 1)
        pooled_logits = (
            view_weights * view_logits
        ).sum(dim=0) / view_weights.sum(dim=0)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, :]
        ).float().mean(dim=0)
        agreement_uncertainty = view_agreement * (1.0 - view_agreement)
        calibration = 1.22775 * (
            0.92211476
            + 0.07788524 * view_agreement
            - 0.02 * agreement_uncertainty
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE