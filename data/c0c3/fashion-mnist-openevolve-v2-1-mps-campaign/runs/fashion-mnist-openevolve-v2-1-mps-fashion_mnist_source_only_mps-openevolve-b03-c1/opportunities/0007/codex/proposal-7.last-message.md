MECHANISM: Full-support translation-and-flip test-time ensembling

HYPOTHESIS: Averaging all 50 views from the training augmentation distribution will exceed 9,082 correct predictions by further reducing translation-sensitive errors.

INTENDED_EDIT: Replace the ten-view cross-shaped ensemble with every combination of ±2-pixel horizontal and vertical translation, each with its horizontal flip.

EVIDENCE: Ten-view translation-and-flip inference improved accuracy from 9,050 to 9,073, and per-image sampling across the full ±2 training range then improved it to 9,082; evaluating the complete matched transformation distribution directly extends both gains.

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        logits = []
        for view in views:
            logits.append(self._forward_once(view))
            logits.append(self._forward_once(view.flip(-1)))
        return torch.stack(logits, dim=0).mean(dim=0)
=======
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        logits = []
        for offset_y in range(5):
            for offset_x in range(5):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                logits.append(self._forward_once(view))
                logits.append(self._forward_once(view.flip(-1)))
        return torch.stack(logits, dim=0).mean(dim=0)
>>>>>>> REPLACE