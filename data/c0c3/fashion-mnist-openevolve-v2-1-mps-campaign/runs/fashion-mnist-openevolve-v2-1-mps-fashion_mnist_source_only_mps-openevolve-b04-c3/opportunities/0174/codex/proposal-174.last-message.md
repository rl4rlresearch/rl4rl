MECHANISM: Lower-live EMA endpoint blending

HYPOTHESIS: Reducing the live-model contribution from 25% to 22.5% will exceed 9,251 correct predictions, or preserve 9,251 while lowering cross-entropy, because increasing it to 27.5% reduced correctness to 9,249 and worsened cross-entropy.

INTENDED_EDIT: Change the validation probability ensemble from 75/25 to 77.5/22.5 EMA/live while leaving training, runtime, and calibration unchanged.

EVIDENCE: The verified 75/25 blend achieved 9,251 correct, whereas 72.5/27.5 achieved only 9,249; this symmetric lower-side test directly evaluates whether the EMA prediction is more reliable than the noisy final endpoint.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.775),
                    F.log_softmax(live_logits, dim=1) + math.log(0.225),
                )
>>>>>>> REPLACE