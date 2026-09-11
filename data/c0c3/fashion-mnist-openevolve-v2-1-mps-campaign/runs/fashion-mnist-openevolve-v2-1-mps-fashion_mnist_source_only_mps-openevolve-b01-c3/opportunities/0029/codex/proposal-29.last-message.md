MECHANISM: Recency-weighted tail parameter averaging

HYPOTHESIS: Linearly weighting later iterates within the proven final-10% averaging window will exceed 9,282 correct predictions by preserving its beneficial temporal coverage while making the installed weights more consistent with terminal BatchNorm statistics.

INTENDED_EDIT: Keep the verified architecture, augmentation, schedule, and averaging window, but replace uniform tail averaging with a linear recency-weighted mean at identical computational cost.

EVIDENCE: Final-10% uniform averaging achieved the best result at 9,282 correct, outperforming final-5% averaging at 9,275; meanwhile BatchNorm-alignment and endpoint-blending attempts timed out, motivating a zero-overhead recency bias within the successful window.

<<<<<<< SEARCH
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
        parameters = [
=======
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
        parameters = [
>>>>>>> REPLACE