MECHANISM: Learned low-rank channel-covariance augmentation

HYPOTHESIS: At the verified beta2=0.96 and 2.0e-3 learning rate, explicitly adding translation-tolerant second-order channel statistics while preserving the spatial flattening path will exceed 9,283 correct predictions by improving discrimination between classes with similar outlines but different part and texture co-occurrences.

INTENDED_EDIT: Restore the best learning rate and augment the 4×4 representation with the unique entries of a centered covariance matrix from a learned 12-channel projection; reduce the classifier width from 140 to 133 to remain below the parameter ceiling.

EVIDENCE: The unmodified representation reached 9,283 correct at 2.0e-3, whereas 2.1e-3 fell to 9,277 and non-local attention fell to 9,265. This tests a different assumption: rather than replacing the proven spatial path or learning unstable pairwise attention, it supplies compact explicit quadratic evidence alongside that path.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
=======
        self.second_order_projection = nn.Conv2d(
            64, 12, kernel_size=1, bias=False
        )
        self.second_order_norm = nn.LayerNorm(78)
        self.register_buffer(
            "second_order_indices", torch.triu_indices(12, 12)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 + 78, 133),
            nn.LayerNorm(133),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(133, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features)
=======
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )

        projected = self.second_order_projection(features).flatten(2)
        projected = projected - projected.mean(dim=2, keepdim=True)
        covariance = torch.bmm(
            projected, projected.transpose(1, 2)
        ) / projected.size(2)
        covariance = covariance[
            :,
            self.second_order_indices[0],
            self.second_order_indices[1],
        ]
        covariance = torch.sign(covariance) * torch.sqrt(
            covariance.abs() + 1.0e-6
        )
        covariance = self.second_order_norm(covariance)

        representation = torch.cat(
            (features.flatten(1), covariance), dim=1
        )
        return self.classifier(representation)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.1e-3,
        betas=(0.9, 0.96),
=======
        lr=2.0e-3,
        betas=(0.9, 0.96),
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.1e-3 * multiplier
=======
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE