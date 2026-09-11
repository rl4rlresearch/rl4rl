MECHANISM: Phase-split temporal pooling with nested causal subsampling

HYPOTHESIS: A 110-unit GRU using 23 frames and separate early/late mean readouts plus the final state will retain at least 85% accuracy while reducing total inference MACs by about 4.1% versus the qualified 24-step design.

INTENDED_EDIT: Use the qualified 110-unit width, omit frame 25 in addition to frames 1 and 27–30, and replace the global mean with separately accumulated first-12-step and remaining-step means.

EVIDENCE: The 110-unit 24-step model qualified at 85.40%, while adding a complementary temporal view previously raised the nested 28-step model from 84.79% to 87.48%; phase-split pooling preserves both coarse temporal order and the final-state view while one fewer recurrent step saves substantially more MACs than its wider classifier adds.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
=======
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(330, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, early_summary, late_summary, count
>>>>>>> REPLACE

<<<<<<< SEARCH
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0
=======
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, early_summary, late_summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        early_weight = (count < 12.0).to(dtype=output.dtype)
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            late_summary + (1.0 - early_weight) * output,
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
=======
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, early_summary, late_summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        positions = count + torch.arange(
            frames.shape[1], device=frames.device, dtype=count.dtype
        ).unsqueeze(0)
        early_weight = (positions < 12.0).to(dtype=outputs.dtype).unsqueeze(2)
        return (
            hidden.transpose(0, 1),
            early_summary + (early_weight * outputs).sum(dim=1),
            late_summary + ((1.0 - early_weight) * outputs).sum(dim=1),
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
=======
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, early_summary, late_summary, count = state
        early_count = count.clamp(max=12.0).clamp_min(1.0)
        late_count = (count - 12.0).clamp_min(1.0)
        early_mean = early_summary / early_count
        late_mean = late_summary / late_count
        return self.classifier(
            torch.cat((early_mean, late_mean, hidden[:, 0, :]), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if available_frames >= 4:
            return [
                frame
                for frame in schedule
                if frame not in (1, available_frames - 2)
            ]
=======
        if available_frames >= 8:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    1,
                    available_frames - 7,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
>>>>>>> REPLACE