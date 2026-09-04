MECHANISM: Reflection-and-translation orbit consistency training

HYPOTHESIS: Training each example with its reflected pair at two neighboring offsets, plus consistency between the offset ensembles, will exceed 9,312 correct predictions by extending the successful flip-paired regularization to the remaining transformations used at validation.

INTENDED_EDIT: Add a cyclic one-pixel translated pair to every training batch, supervise all four views, retain the 5% within-offset flip consistency loss, and add a smaller 2.5% consistency loss between the two flip-averaged offsets.

EVIDENCE: Flip-paired consistency training improved correctness from 9,289 to 9,312, while learned reflection-orbit feature fusion fell to 9,280; this favors preserving the successful architecture and extending consistency training to validation-time offset variation.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_logits = model(torch.cat((images, images.flip(-1)), dim=0))
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=0.02,
        )
    )

    original_log_probabilities = F.log_softmax(original_logits, dim=-1)
    flipped_log_probabilities = F.log_softmax(flipped_logits, dim=-1)
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probabilities,
            flipped_log_probabilities.exp(),
            reduction="batchmean",
        )
        + F.kl_div(
            flipped_log_probabilities,
            original_log_probabilities.exp(),
            reduction="batchmean",
        )
    )
    return classification_loss + 0.05 * consistency_loss
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del total_steps
    height, width = images.shape[-2:]
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    row_offset, col_offset = (
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    )[step % 4]
    translated_images = padded[
        :,
        :,
        row_offset : row_offset + height,
        col_offset : col_offset + width,
    ]

    view_logits = model(
        torch.cat(
            (
                images,
                images.flip(-1),
                translated_images,
                translated_images.flip(-1),
            ),
            dim=0,
        )
    )
    (
        original_logits,
        flipped_logits,
        translated_logits,
        translated_flipped_logits,
    ) = view_logits.chunk(4, dim=0)

    classification_loss = 0.25 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            translated_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            translated_flipped_logits,
            labels,
            label_smoothing=0.02,
        )
    )

    def symmetric_kl(
        first_logits: torch.Tensor,
        second_logits: torch.Tensor,
    ) -> torch.Tensor:
        first_log_probabilities = F.log_softmax(first_logits, dim=-1)
        second_log_probabilities = F.log_softmax(second_logits, dim=-1)
        return 0.5 * (
            F.kl_div(
                first_log_probabilities,
                second_log_probabilities.exp(),
                reduction="batchmean",
            )
            + F.kl_div(
                second_log_probabilities,
                first_log_probabilities.exp(),
                reduction="batchmean",
            )
        )

    flip_consistency_loss = 0.5 * (
        symmetric_kl(original_logits, flipped_logits)
        + symmetric_kl(translated_logits, translated_flipped_logits)
    )
    original_offset_logits = 0.5 * (original_logits + flipped_logits)
    translated_offset_logits = 0.5 * (
        translated_logits + translated_flipped_logits
    )
    translation_consistency_loss = symmetric_kl(
        original_offset_logits,
        translated_offset_logits,
    )
    return (
        classification_loss
        + 0.05 * flip_consistency_loss
        + 0.025 * translation_consistency_loss
    )
>>>>>>> REPLACE