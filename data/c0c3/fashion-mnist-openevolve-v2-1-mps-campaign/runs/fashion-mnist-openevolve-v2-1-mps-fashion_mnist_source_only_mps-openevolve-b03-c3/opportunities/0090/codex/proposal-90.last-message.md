MECHANISM: Class-conditioned spatial evidence pooling

HYPOTHESIS: Replacing the absolute-position flattened MLP with lightweight convolutional relation modeling and class-specific learned pooling will exceed 9,268 correct predictions by letting each class select informative garment parts while retaining distributed silhouette statistics.

INTENDED_EDIT: Preserve the verified multiscale tokenizer, add two efficient 4×4 spatial-relation blocks, and compute logits from class-specific evidence maps, learned attention/uniform mixtures, and global activation statistics. Retain the best verified distance-stratified TTA.

EVIDENCE: Fixed flattening remains the strongest design, but static residual and pooled fusion reached only 9,217 and 9,236; full class-query attention directly challenged this limitation but timed out. This patch tests content-dependent class pooling without quadratic attention, stays below 250,000 parameters, and uses the radius-2 weighting that preserved 9,268 correct while lowering cross-entropy.

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
        self.relation = nn.Sequential(
            nn.Conv2d(64, 144, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(144),
            nn.GELU(),
            nn.Conv2d(
                144, 144, kernel_size=3, padding=1, groups=144, bias=False
            ),
            nn.BatchNorm2d(144),
            nn.GELU(),
            nn.Conv2d(144, 144, kernel_size=1, bias=False),
            nn.BatchNorm2d(144),
        )
        self.relation_refine = nn.Sequential(
            nn.Conv2d(
                144, 144, kernel_size=3, padding=1, groups=144, bias=False
            ),
            nn.BatchNorm2d(144),
            nn.GELU(),
            nn.Conv2d(144, 144, kernel_size=1, bias=False),
            nn.BatchNorm2d(144),
        )
        self.class_evidence = nn.Conv2d(144, 10, kernel_size=1)
        self.class_attention = nn.Conv2d(144, 10, kernel_size=1)
        self.attention_mix = nn.Parameter(torch.zeros(10))
        self.global_classifier = nn.Linear(2 * 144, 10)
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
        features = self.relation(features)
        features = F.gelu(features + self.relation_refine(features))

        evidence = self.class_evidence(features).flatten(2)
        attention = self.class_attention(features).flatten(2).softmax(dim=-1)
        attention_mix = torch.sigmoid(self.attention_mix)[None, :, None]
        pooling_weights = (
            attention_mix * attention
            + (1.0 - attention_mix) / evidence.size(-1)
        )
        spatial_logits = (pooling_weights * evidence).sum(dim=-1)

        global_mean = features.mean(dim=(2, 3))
        global_rms = torch.sqrt(
            features.square().mean(dim=(2, 3)) + 1.0e-6
        )
        global_logits = self.global_classifier(
            torch.cat((global_mean, global_rms), dim=1)
        )
        return spatial_logits + global_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
=======
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE