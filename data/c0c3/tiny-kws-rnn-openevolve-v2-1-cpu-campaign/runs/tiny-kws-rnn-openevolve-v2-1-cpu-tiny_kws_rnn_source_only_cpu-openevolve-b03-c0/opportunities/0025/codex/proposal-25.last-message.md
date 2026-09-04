MECHANISM: Incremental adjacent-band spectral pooling

HYPOTHESIS: Averaging mel bands 16–17 in addition to the already pooled bands 18–19 will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.85%.

INTENDED_EDIT: Reduce the normalized GRU input from 19 to 18 features using two separate adjacent-band averages for the four highest mel bands.

EVIDENCE: Pooling bands 18–19 with the seven-output zero-sum head passed at 85.52%; this tests one additional structural input reduction while preserving more high-frequency detail than averaging all four bands together.

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
                frame[:, :16],
                frame[:, 16:18].mean(dim=1, keepdim=True),
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
                frames[:, :, :16],
                frames[:, :, 16:18].mean(dim=2, keepdim=True),
                frames[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
>>>>>>> REPLACE