MECHANISM: Batch-aligned hard-maximum attention

HYPOTHESIS: Using 125-example batches with hard-maximum channel saliency will exceed 9,322 correct predictions by providing 800 uniform optimizer steps without partial batches while retaining the faster, lower-cross-entropy attention design.

INTENDED_EDIT: Replace top-four saliency with global maximum attention and set the batch size to 125, which exactly divides each 50,000-example training pass.

EVIDENCE: Hard-maximum attention achieved 9,320 correct with lower cross-entropy and 3.5 seconds less training time than top-four; batch size 125 removes the two 80-example tail batches and adds 18 full optimizer updates within that timing margin.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 125
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
>>>>>>> REPLACE