MECHANISM: Progressive upper-skip adjacent-band folding

HYPOTHESIS: Folding normalized upper-skip bands 12–13 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.

INTENDED_EDIT: Reduce the upper GRU input width from 65 to 64 and replace its 16-feature acoustic skip with the first 12 normalized bands plus averages of bands 12–13, 14–15, and 16–17 in both recurrent paths.

EVIDENCE: The two preceding upper-skip folds each preserved at least 85.64% accuracy while saving 3,286,080 recurrent MACs; repeating the same localized compression on the next adjacent pair is the narrowest evidence-supported reduction.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(65, 48, num_layers=1, batch_first=True)
=======
        self.upper_gru = nn.GRU(64, 48, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
=======
        upper_acoustic = torch.cat(
            (
                normalized[..., :12],
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
=======
        upper_acoustic = torch.cat(
            (
                normalized[..., :12],
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
>>>>>>> REPLACE