MECHANISM: Relaxed confidence-gated penultimate exit

HYPOTHESIS: Lowering the qualified penultimate exit threshold from 0.90 to 0.85 will preserve validation accuracy at or above 85% while reducing total inference MACs below 254,608,200.

INTENDED_EDIT: Adopt the qualified 22-frame schedule, non-affine normalization, and bias-free seven-logit head; avoid classifier MACs before the penultimate step; and exit examples with at least 0.85 posterior confidence after step 21.

EVIDENCE: Lowering the threshold from 0.95 to 0.90 increased early exits, reduced total MACs from 256,362,120 to 254,608,200, and retained 85.40% accuracy; the small observed accuracy change motivates another measured relaxation.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
        hidden, summary, maximum, count = state
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]

    def exit_mask(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        count = state[3][:, 0]
        confidence = logits.softmax(dim=1).amax(dim=1)
        return (count == float(total_steps - 1)) & (confidence >= 0.85)
>>>>>>> REPLACE