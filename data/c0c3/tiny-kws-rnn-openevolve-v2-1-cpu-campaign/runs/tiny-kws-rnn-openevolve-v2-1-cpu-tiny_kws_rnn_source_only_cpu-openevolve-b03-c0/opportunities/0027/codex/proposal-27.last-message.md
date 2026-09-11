MECHANISM: Dual-edge adjacent-band spectral pooling

HYPOTHESIS: Averaging mel bands 0–1 while retaining the successful 18–19 average will achieve at least 85% validation accuracy at the lower 18-input inference cost, because it preserves bands 2–17 that were damaged by both failed 18-input designs.

INTENDED_EDIT: Reduce the GRU input from 19 to 18 features by separately averaging the lowest and highest mel-band pairs while preserving every middle band.

EVIDENCE: The 19-input model pooling bands 18–19 passed at 85.52%, while further pooling bands 16–17 scored 84.29% and pooling bands 17–19 scored 84.54%; this motivates moving the additional compression away from bands 16–17.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 98, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 98, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_frame = torch.cat(
            (frame[:, :18], frame[:, 18:].mean(dim=1, keepdim=True)), dim=1
        )
=======
        pooled_frame = torch.cat(
            (
                frame[:, :2].mean(dim=1, keepdim=True),
                frame[:, 2:18],
                frame[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_frames = torch.cat(
            (frames[:, :, :18], frames[:, :, 18:].mean(dim=2, keepdim=True)), dim=2
        )
=======
        pooled_frames = torch.cat(
            (
                frames[:, :, :2].mean(dim=2, keepdim=True),
                frames[:, :, 2:18],
                frames[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
>>>>>>> REPLACE