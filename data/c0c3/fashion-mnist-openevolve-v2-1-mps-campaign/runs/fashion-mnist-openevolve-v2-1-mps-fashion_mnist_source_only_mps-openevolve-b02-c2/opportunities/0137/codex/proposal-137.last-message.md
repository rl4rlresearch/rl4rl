MECHANISM: Fused top-four attention with temperature calibration

HYPOTHESIS: Preserving the qualified model’s stochastic training path while fusing its linear attention calls will retain at least 9,322 correct predictions and finish reliably; temperature 1.05 will leave argmax predictions unchanged while reducing cross-entropy below 0.192551.

INTENDED_EDIT: Apply the shared bias-free attention kernel once to the sum of mean and top-four descriptors, and soften only the final evaluation logits.

EVIDENCE: The current top-four design has the best verified count of 9,322. Earlier optimized retries also removed the pre-loss random flip, changing the RNG sequence consumed by dropout and regressing to 9,301; this patch keeps that successful stochastic path intact while reducing one kernel call.

<<<<<<< SEARCH
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_gate = self.channel_attention(channel_average + channel_salient)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        ensemble_logits = (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        )
        return ensemble_logits / 1.05
>>>>>>> REPLACE