MECHANISM: Hierarchical flip-logit and translation-probability ensemble

HYPOTHESIS: Averaging each original/flip pair in logit space before averaging the five position-level probabilities will exceed 9,112 correct predictions by preserving the proven flip consensus while isolating probability averaging to the less reliable translated views.

INTENDED_EDIT: Keep the validated architecture and training recipe, but evaluate centered and four cardinal translations with hierarchical aggregation: logit averaging within each flip pair, then probability averaging across positions.

EVIDENCE: Flip-logit averaging improved the residual model from 9,086 to 9,094 correct, while cardinal probability averaging reached the best result of 9,112; combining those successful aggregation rules according to augmentation type is the next direct test.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.mid(self.stem(images))
        features = F.gelu(features + self.late2(self.late1(features)))
        logits = self.classifier(features)
        if self.training:
            return logits

        flipped = images.flip(-1)
        flipped_features = self.mid(self.stem(flipped))
        flipped_features = F.gelu(
            flipped_features + self.late2(self.late1(flipped_features))
        )
        flipped_logits = self.classifier(flipped_features)
        return 0.5 * (logits + flipped_logits)
=======
    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.mid(self.stem(images))
        features = F.gelu(features + self.late2(self.late1(features)))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._predict(images)

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
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
>>>>>>> REPLACE