MECHANISM: Native-alignment-matched translation marginalization

HYPOTHESIS: Sampling and ensembling the centered view with 50% probability will exceed 9,285 correct predictions by preserving translation robustness while reducing dilution of the validation images’ native alignment.

INTENDED_EDIT: Give the centered crop half the training and inference probability mass, with the four cardinal shifts receiving one eighth each; retain the proven architecture and final-10% linear recency average.

EVIDENCE: The strongest verified design reached 9,285 correct with cardinal translations and linear tail averaging, while the inference-only center-weighting attempt timed out without accuracy evidence; matching both training and inference distributions tests the native-alignment hypothesis without additional computation.

<<<<<<< SEARCH
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
=======
        position_logits = view_logits.mean(dim=1)
        position_weights = position_logits.new_tensor(
            (0.5, 0.125, 0.125, 0.125, 0.125)
        )
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1)
            + position_weights.log()[:, None, None],
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
=======
    position_codes = torch.randint(0, 8, (batch,), device=images.device)
    positions = torch.where(
        position_codes < 4,
        torch.zeros_like(position_codes),
        position_codes - 3,
    )
    row_offsets = torch.tensor(
>>>>>>> REPLACE