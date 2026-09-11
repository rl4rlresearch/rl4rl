MECHANISM: Zero-sum-head-assisted adjacent-band spectral pooling

HYPOTHESIS: Combining the final two mel bands into one input while retaining the seven-output zero-sum head will achieve at least 85% validation accuracy and reduce recurrent MACs by about 0.85%.

INTENDED_EDIT: Reduce the GRU input and input normalization from 20 to 19 features by averaging the two highest mel bands at every recurrent step.

EVIDENCE: The prior 19-input averaged-band model narrowly missed at 84.91%, while the seven-output zero-sum head improved the 97-coordinate design from 85.52% to 86.26% and enabled the current 96-coordinate design to pass at 85.28%.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 98, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        pooled_frame = torch.cat(
            (frame[:, :18], frame[:, 18:].mean(dim=1, keepdim=True)), dim=1
        )
        output, hidden = self.gru(
            self.input_norm(pooled_frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        pooled_frames = torch.cat(
            (frames[:, :, :18], frames[:, :, 18:].mean(dim=2, keepdim=True)), dim=2
        )
        outputs, hidden = self.gru(
            self.input_norm(pooled_frames), hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE