MECHANISM: Dual-statistic channel attention

HYPOTHESIS: Restoring shared global-average and global-maximum channel evidence will raise validation correctness from 9,286 toward the verified 9,320 result without adding parameters or costly dispersion computation.

INTENDED_EDIT: Replace average-only channel attention with the verified parameter-neutral average-plus-maximum gate while preserving paired-view supervision and flip-ensemble inference.

EVIDENCE: The identical 249,855-parameter regimen achieved 9,320 correct with average-plus-maximum attention versus 9,286 with average-only attention; adding standard deviation later exceeded the verification time limit.

<<<<<<< SEARCH
        channel_summary = F.adaptive_avg_pool2d(features, 1)
        channel_summary = channel_summary.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_summary)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE