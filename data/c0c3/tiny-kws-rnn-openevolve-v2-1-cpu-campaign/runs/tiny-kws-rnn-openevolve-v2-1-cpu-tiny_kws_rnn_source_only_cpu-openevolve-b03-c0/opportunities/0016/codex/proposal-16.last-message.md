MECHANISM: Temporal multiplexing of adjacent high-frequency mel bands

HYPOTHESIS: Alternating the final GRU input between mel bands 18 and 19 will preserve their distinct spectral cues and achieve at least 85% accuracy while retaining the 19-input model’s lower 897,738,800-MAC inference cost.

INTENDED_EDIT: Keep the passing 98-unit, 32-step architecture, reduce its GRU input to 19 features, and alternate which of the two highest mel bands occupies the final input channel at each causal step.

EVIDENCE: Averaging the two highest bands reduced inference MACs by 0.85% and narrowly missed accuracy at 84.91% with better cross-entropy than the passing baseline; temporal multiplexing tests whether retaining their distinct values avoids the information loss from averaging.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0
=======
        hidden, summary, count = state
        use_last_band = count[:, 0].to(torch.long).remainder(2).bool()
        high_band = torch.where(use_last_band, frame[:, 19], frame[:, 18])
        compressed = torch.cat((frame[:, :18], high_band.unsqueeze(1)), dim=1)
        output, hidden = self.gru(
            self.input_norm(compressed).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        hidden, summary, count = state
        offsets = torch.arange(frames.shape[1], device=frames.device).unsqueeze(0)
        use_last_band = (
            count.to(torch.long) + offsets
        ).remainder(2).bool()
        high_band = torch.where(
            use_last_band, frames[:, :, 19], frames[:, :, 18]
        )
        compressed = torch.cat(
            (frames[:, :, :18], high_band.unsqueeze(2)), dim=2
        )
        outputs, hidden = self.gru(
            self.input_norm(compressed), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
>>>>>>> REPLACE