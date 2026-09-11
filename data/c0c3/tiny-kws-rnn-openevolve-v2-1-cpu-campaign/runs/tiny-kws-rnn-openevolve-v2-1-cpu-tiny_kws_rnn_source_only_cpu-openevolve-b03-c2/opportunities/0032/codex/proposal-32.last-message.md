MECHANISM: Expanded penultimate-frame confidence exit

HYPOTHESIS: Lowering the qualified early-exit threshold from 0.98 to 0.95 will preserve validation_accuracy >= 0.85 while reducing total_inference_macs below 764,015,400 by skipping the final recurrent step for more stable predictions.

INTENDED_EDIT: Restore the qualified 110-unit, 22-frame, 219-feature bias-free model with LayerNorm bias removed, and broaden its penultimate-step early exit to confidence >= 0.95.

EVIDENCE: The 0.98 exit model matched the no-exit model’s 85.276% accuracy exactly while eliminating 180 recurrent steps, indicating headroom to admit additional slightly lower-confidence penultimate predictions.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.input_norm = nn.LayerNorm(20, bias=False)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
        hidden, summary, count = state
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-1]), dim=1)
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 6:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    0,
                    1,
                    2,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
        return schedule

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        penultimate = count[:, 0] == float(total_steps - 1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        return penultimate & (confidence >= 0.95)
>>>>>>> REPLACE