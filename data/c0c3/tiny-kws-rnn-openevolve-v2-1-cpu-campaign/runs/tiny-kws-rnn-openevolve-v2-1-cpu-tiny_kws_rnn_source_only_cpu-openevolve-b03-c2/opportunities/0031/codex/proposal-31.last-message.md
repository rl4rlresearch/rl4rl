MECHANISM: Confidence-gated penultimate-frame exit

HYPOTHESIS: Exiting examples whose penultimate-step confidence is at least 0.98 will preserve validation_accuracy >= 0.85 and reduce total_inference_macs below 770,624,880 by avoiding the final recurrent step for enough stable predictions.

INTENDED_EDIT: Restore the qualified 110-unit, 22-step, 219-feature bias-free design with LayerNorm bias removed, then add a penultimate-step early exit; intermediate classifier calls use recurrent hidden logits without executing the dense readout.

EVIDENCE: Reference Design 3 achieved 85.28% at 22 steps and 770,624,880 MACs, while an unconditional alternative 22-step frame removal scored 84.54%; selectively skipping the isolated final frame only for already-confident examples is a lower-risk route to fewer MACs than another universal frame deletion.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
=======
        self.input_norm = nn.LayerNorm(20, bias=False)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 4:
            return [
                frame
                for frame in schedule
                if frame not in (1, available_frames - 2)
            ]
        return schedule
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
        return penultimate & (confidence >= 0.98)
>>>>>>> REPLACE