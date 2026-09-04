MECHANISM: Center-weighted translation ensemble with linear tail averaging

HYPOTHESIS: Giving the native centered view half of the inference probability mass while restoring final-10% linear recency averaging will exceed 9,285 correct predictions by retaining translation robustness without letting four shifted crops overwhelm the validation image’s true alignment.

INTENDED_EDIT: Restore the strongest verified final-10% linearly weighted parameter average and change inference from uniform position averaging to 50% centered and 12.5% per cardinal shift.

EVIDENCE: Final-10% linear recency averaging achieved the best verified result of 9,285 correct; cardinal translations were beneficial, while the unverified center-biased training proposal identified native-alignment dilution as a plausible remaining issue. Weighting only inference isolates that idea without additional training cost.

<<<<<<< SEARCH
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
=======
        position_logits = view_logits.mean(dim=1)
        log_position_weights = position_logits.new_tensor(
            (0.5, 0.125, 0.125, 0.125, 0.125)
        ).log()
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1)
            + log_position_weights[:, None, None],
            dim=0,
        )
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