MECHANISM: Runtime-trimmed stronger disagreement-gated flip fusion

HYPOTHESIS: The reported 0.0275 adjustment will retain 9,331 correct predictions, reduce cross-entropy below 0.1962260711669922, and complete verification after eliminating redundant evaluation-time argmax operations.

INTENDED_EDIT: Increase the adaptive adjustment from 0.025 to 0.0275 and reuse top-k indices to detect view disagreement.

EVIDENCE: A prior 0.0275 run reported 9,331 correct and 0.1962239959716797 cross-entropy; its failure was attributed to runtime rather than validation quality.

<<<<<<< SEARCH
            original_top2 = logits.topk(2, dim=1).values
            flipped_top2 = flipped_logits.topk(2, dim=1).values
            original_margin = original_top2[:, 0] - original_top2[:, 1]
            flipped_margin = flipped_top2[:, 0] - flipped_top2[:, 1]
            margin_difference = flipped_margin - original_margin

            disagreement = logits.argmax(dim=1) != flipped_logits.argmax(dim=1)
            decisive = disagreement & (margin_difference.abs() > 0.5)
            adjustment = (
                0.025
                * torch.tanh(0.5 * margin_difference)
                * decisive.to(logits.dtype)
            )
=======
            original_top2 = logits.topk(2, dim=1)
            flipped_top2 = flipped_logits.topk(2, dim=1)
            original_margin = (
                original_top2.values[:, 0] - original_top2.values[:, 1]
            )
            flipped_margin = (
                flipped_top2.values[:, 0] - flipped_top2.values[:, 1]
            )
            margin_difference = flipped_margin - original_margin

            disagreement = (
                original_top2.indices[:, 0] != flipped_top2.indices[:, 0]
            )
            decisive = disagreement & (margin_difference.abs() > 0.5)
            adjustment = (
                0.0275
                * torch.tanh(0.5 * margin_difference)
                * decisive.to(logits.dtype)
            )
>>>>>>> REPLACE