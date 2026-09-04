MECHANISM: EMA–endpoint snapshot probability ensemble

HYPOTHESIS: Blending the stronger 0.99 EMA prediction with 25% of the final live-model prediction will exceed 9,247 correct predictions, or tie while lowering validation cross-entropy.

INTENDED_EDIT: During validation, compute calibrated flip ensembles for both EMA and live parameters, then combine their class probabilities with 0.75/0.25 weights.

EVIDENCE: The 0.99 EMA improved validation_correct from 9,243 to 9,247, while calibration changes alone could not alter rankings; snapshot blending extends the only successful ranking-changing mechanism without changing training or learned-parameter count.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            power = 0.5831695556640625
            logits = 1.03592 * (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            power = 0.5831695556640625
            ema_logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power

            if self._using_ema:
                self._swap_ema_state()
                try:
                    live_logits = self._classify(images)
                    live_flipped_logits = self._classify(images.flip(-1))
                    live_logits = (
                        torch.logaddexp(
                            power * F.log_softmax(live_logits, dim=1),
                            power * F.log_softmax(live_flipped_logits, dim=1),
                        )
                        - math.log(2.0)
                    ) / power
                finally:
                    self._swap_ema_state()

                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
            else:
                logits = ema_logits
            logits = 1.03592 * logits
        return logits
>>>>>>> REPLACE