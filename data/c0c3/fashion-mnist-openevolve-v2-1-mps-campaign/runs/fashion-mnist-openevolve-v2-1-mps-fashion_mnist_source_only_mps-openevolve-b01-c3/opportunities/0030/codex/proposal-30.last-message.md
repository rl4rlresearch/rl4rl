MECHANISM: Coarse spatial-pyramid classification head

HYPOTHESIS: Preserving 2×2 feature layout alongside global mean and peak statistics will exceed 9,285 correct predictions because it exposes class-relevant spatial arrangement that the shared globally invariant readout discards.

INTENDED_EDIT: Replace the global-only classifier input with a 2×2 spatial pyramid plus global summaries, resize its bottleneck to remain under 250,000 parameters, and evaluate it using the proven final-10% recency-weighted averaging procedure.

EVIDENCE: Local spatial refinement improved correctness from 9,202 to 9,209, and position-matched augmentation reached 9,262, indicating that spatial structure matters; however, all available designs still erase feature layout with mean/max pooling. The 9,285-result establishes recency-weighted averaging as the strongest training baseline for testing this alternative representation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.LayerNorm(576),
            nn.Linear(576, 19),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(19, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        spatial_features = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        pooled_features = torch.cat(
            (spatial_features, mean_features, peak_features),
            dim=1,
        )
        return self.classifier(pooled_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.85 * total_steps))
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
>>>>>>> REPLACE

<<<<<<< SEARCH
    tail_distance = total_steps - step - 1
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 3 != 2
    ):
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
=======
    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
>>>>>>> REPLACE