MECHANISM: Full-exposure train–inference view matching

HYPOTHESIS: Training uniformly on the five inference views for all 100,000 examples will exceed 9,214 correct predictions by eliminating the current center-view overrepresentation without adding model-forward computation.

INTENDED_EDIT: Extend the deterministic center-and-cardinal crop cycle through the entire training budget instead of switching to clean-only images halfway through.

EVIDENCE: The current matched-view design achieved 9,214 correct, while prior evidence found shifted ten-view inference beneficial. Currently 60% of training examples are centered although inference weights all five spatial views equally; this patch makes both distributions 20% per view.

<<<<<<< SEARCH
    if step < total_steps // 2:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        view = step % 5
        top = (1, 0, 2, 1, 1)[view]
        left = (1, 1, 1, 0, 2)[view]
        images = padded[:, :, top : top + 28, left : left + 28]
    return images, labels
=======
    del total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    view = step % 5
    top = (1, 0, 2, 1, 1)[view]
    left = (1, 1, 1, 0, 2)[view]
    images = padded[:, :, top : top + 28, left : left + 28]
    return images, labels
>>>>>>> REPLACE