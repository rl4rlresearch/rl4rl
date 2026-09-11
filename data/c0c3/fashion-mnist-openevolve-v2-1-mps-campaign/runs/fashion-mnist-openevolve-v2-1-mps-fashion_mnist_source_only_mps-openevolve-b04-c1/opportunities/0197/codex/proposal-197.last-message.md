MECHANISM: Stronger temporal averaging for cross-view fusion

HYPOTHESIS: Lowering only the view-fusion EMA rate from 0.02 to 0.015 will exceed 9,330 correct predictions by suppressing harmful late drift in the invariant/disagreement projection while preserving the proven classifier and backbone averaging rates.

INTENDED_EDIT: Use EMA rate 0.015 for `view_fusion` parameters, 0.04 for classifier parameters, and 0.02 for all remaining learned parameters.

EVIDENCE: Increasing the view-fusion EMA rate reduced correctness from 9,330 to 9,326, while anchoring its invariant path was motivated by the same projection-drift concern; testing stronger averaging is the direct unexplored opposite direction without changing the established architecture.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.015
                            if name.startswith("view_fusion.")
                            else 0.02
                        )
>>>>>>> REPLACE