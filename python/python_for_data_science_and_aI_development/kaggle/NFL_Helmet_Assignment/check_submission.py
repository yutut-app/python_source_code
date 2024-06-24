def check_submission(sub):
    """
    Checks that the submission meets all the requirements.

    1. No more than 22 Boxes per frame.
    2. Only one label prediction per video/frame
    3. No duplicate boxes per frame.

    Returns:
        True -> Passed the tests
        False -> Failed the test
    """
    # Maximum of 22 boxes per frame.
    max_box_per_frame = sub.groupby(["video_frame"])["label"].count().max()
    if max_box_per_frame > 22:
        print("Has more than 22 boxes in a single frame")
        return False
    # Only one label allowed per frame.
    has_duplicate_labels = sub[["video_frame", "label"]].duplicated().any()
    if has_duplicate_labels:
        print("Has duplicate labels")
        return False
    # Check for unique boxes
    has_duplicate_boxes = (
        sub[["video_frame", "left", "width", "top", "height"]].duplicated().any()
    )
    if has_duplicate_boxes:
        print("Has duplicate boxes")
        return False
    return True