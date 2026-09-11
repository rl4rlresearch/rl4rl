MECHANISM: Canonical paired-view training

HYPOTHESIS: Removing the redundant preparatory flip will retain at least 9,322 correct predictions while reducing overhead enough for the verified top-four attention design to finish reliably.

INTENDED_EDIT: Pass canonical images directly to the loss; paired supervision still trains on every image and its horizontal flip.

EVIDENCE: Exact top-four saliency achieved the best verified result of 9,322 correct, but subsequent runs repeatedly timed out; the preparatory flip only swaps the order of the two views later constructed by training_loss.

<<<<<<< SEARCH
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE