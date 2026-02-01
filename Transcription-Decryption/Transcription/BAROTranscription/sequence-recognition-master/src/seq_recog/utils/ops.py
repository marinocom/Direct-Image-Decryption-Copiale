"""Nice function and operator collection."""

import math

import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple, Union


def iou(ref, cmp, thresh=None):
    """Intersection over Union computation.

    Returns the Intersection over Union of a given set of "cmp" grid-aligned
    rectangles against a set of reference "ref" grid-aligned rectangles.
    IoU values below "thresh" will be pulled to 0. If Thresh is None, results
    will be provided as is.

    Parameters
    ----------
    ref : ArrayLike
        Set of grid-aligned rectangles to compare against. Provided as a Nx4
        matrix of N points of (x1, y1, x2, y2) coordinates.
    cmp : ArrayLike
        Set of grid-aligned rectangles to be compared. Provided as a Mx4
        matrix of M points of (x1, y1, x2, y2) coordinates.
    thresh : float, optional
        A threshold value for the final intersection over union. The default
        is None.

    Returns
    -------
    ArrayLike:
        A NxM matrix with the IoU of each cmp rectangle against each reference
        rectangle
    """
    N, M = ref.shape[0], cmp.shape[0]
    s_ref = np.stack([ref] * M, axis=1)
    s_cmp = np.stack([cmp] * N, axis=0)

    # Intersection
    intr_x = np.min(
        np.stack((s_ref[:, :, 2], s_cmp[:, :, 2]), axis=0), axis=0
    ) - np.max(np.stack((s_ref[:, :, 0], s_cmp[:, :, 0]), axis=0), axis=0)
    intr_x = np.maximum(intr_x, 0)

    intr_y = np.min(
        np.stack((s_ref[:, :, 3], s_cmp[:, :, 3]), axis=0), axis=0
    ) - np.max(np.stack((s_ref[:, :, 1], s_cmp[:, :, 1]), axis=0), axis=0)
    intr_y = np.maximum(intr_y, 0)

    intr_t = intr_x * intr_y

    # Union
    area_r = (s_ref[:, :, 2] - s_ref[:, :, 0]) * (s_ref[:, :, 3] - s_ref[:, :, 1])
    area_c = (s_cmp[:, :, 2] - s_cmp[:, :, 0]) * (s_cmp[:, :, 3] - s_cmp[:, :, 1])

    union = area_r + area_c - intr_t

    iou = intr_t / union

    if thresh is not None:
        iou = np.where(iou >= thresh, iou, 0)

    return iou


def fiou(bboxes1, bboxes2):
    """Fast IoU Implementation.

    Fast IoU implementation from https://medium.com/@venuktan/vectorized-intersection-
    over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    We compared it against our own and decided to use this as it is much more
    memory efficient.

    Parameters
    ----------
    bboxes1: ArrayLike
        Array of bounding boxes in XYXY format.
    bboxes2: ArrayLike
        Array of bounding boxes in XYXY format.

    Returns
    -------
    ArrayLike
        Matrix with IoU results of all bounding boxes from the first array against all
        bounding boxes from the second array.
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    xa = np.maximum(x11, np.transpose(x21))
    ya = np.maximum(y11, np.transpose(y21))
    xb = np.minimum(x12, np.transpose(x22))
    yb = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum((xb - xa + 1), 0) * np.maximum((yb - ya + 1), 0)

    boxa_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxb_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = inter_area / (boxa_area + np.transpose(boxb_area) - inter_area)

    return iou


def fiou_1d(bboxes1, bboxes2):
    """Intersection over Union for sequences of 1D boxes (segments).

    :param bboxes1: Matrix of coordinates with shape P x 2 corresponding to the
    prediction.
    :param bboxes2: Matrix of coordinates with shape Q x 2 corresponding to the
    ground truth.
    :returns: Intersection over union of the two sets of boxes.
    """
    x11, x12 = np.split(bboxes1, 2, axis=-1)
    x21, x22 = np.split(bboxes1, 2, axis=-1)

    xa = np.maximum(x11, x21.T)
    xb = np.minimum(x12, x22.T)

    intersection = np.maximum((xb - xa + 1), 0)

    segm_length_1 = x12 - x11
    segm_length_2 = x22 - x21

    iou = intersection / (segm_length_1 + segm_length_2.T - intersection)

    return iou


def seqiou(bboxes1, bboxes2):
    """Intersection over Union for sequences of 1D boxes (segments).

    :param bboxes1: Matrix of coordinates with shape P x 2 corresponding to the
    prediction.
    :param bboxes2: Matrix of coordinates with shape Q x 2 corresponding to the
    ground truth.
    :returns: Intersection over union of the two sets of boxes compared using
    input sequential ordering.
    """
    assert len(bboxes1) == len(bboxes2), "Number of input bboxes is not equal"
    x11, x12 = np.split(bboxes1, 2, axis=-1)
    x21, x22 = np.split(bboxes2, 2, axis=-1)

    x11 = x11.flatten()
    x12 = x12.flatten()
    x21 = x21.flatten()
    x22 = x22.flatten()

    xa = np.maximum(x11, x21)
    xb = np.minimum(x12, x22)

    intersection = np.maximum((xb - xa + 1), 0)

    segm_length_1 = x12 - x11
    segm_length_2 = x22 - x21

    iou = intersection / (segm_length_1 + segm_length_2 - intersection)

    return iou


def sequiou_multiple(bbox_set: ArrayLike) -> ArrayLike:
    """Perform iou of multiple sets of boxes.

    Parameters
    ----------
    bbox_set: ArrayLike
        A set of bounding boxes with shape N x L x 2 where N is the number of
        predictions and L is the sequence length.

    Returns
    -------
    ArrayLike:
        An L-length array with the IoU of each bounding box.
    """
    n, l, _ = bbox_set.shape
    x1, x2 = np.split(bbox_set, 2, axis=-1)
    x1, x2 = x1.squeeze(-1), x2.squeeze(-1)

    x1max = np.max(x1, axis=0)
    x1min = np.min(x1, axis=0)
    x2max = np.max(x2, axis=0)
    x2min = np.min(x2, axis=0)

    intersection = x2min - x1max
    union = x2max - x1min
    union[union <= 0] = 1

    return intersection / union


def compare_gt(iou, confidence):
    """Comparison of ground truth boxes.

    Compare each ground truth box against a prediction and align them. Solve
    conflicts with area of overlap first, then confidence.

    Parameters
    ----------
    iou : ArrayLike
        NxM matrix with the Intersection over Union value of each ground truth
        box (N) against each predicted box (M).
    confidence : ArrayLike
        M-length array with prediction confidence values.

    Returns
    -------
    belonging : ArrayLike
        M-length array with the corresponding ground truth box to each
        predicted box.
    """
    # Maximal iou values are the ones we're interested in
    max_iou = np.max(iou, axis=0)

    # Get the most overlapping gt box for each prediction
    belonging = np.argmax(iou, axis=0)
    belonging = np.where(max_iou > 0, belonging, -1)

    # Find duplicate predictions and ignore non-valid boxes
    vals, counts = np.unique(belonging, return_counts=True)
    counts = counts[vals != -1]
    vals = vals[vals != -1]
    conflicts = np.nonzero(counts > 1)[0]

    # Solve each conflict with non-maximal suppression
    for ii in range(len(conflicts)):
        # Conflicting gt box
        conflict = vals[conflicts[ii]]
        points_conflict = belonging == conflict
        losers = np.nonzero(
            (confidence < np.max(confidence[points_conflict]))
            & (points_conflict)
            & (max_iou > 0)
        )
        belonging[losers] = -1
    return belonging


def nonmax_supression(boxes, confidence, labels, confthresh=0.6, iouthresh=0.5):
    """Remove redundant bounding boxes.

    Removes overlapping redundant boxes according to confidence values. Labels,
    boxes and confidence are returned to keep the data aligned.

    Parameters
    ----------
    boxes : ArrayLike
        Array of N samples in [x1, y1, x2, y2] format
    confidence : ArrayLike
        Array of N samples with a float [0..1] confidence value.
    labels : ArrayLike
        Array of N samples with an integer class value.
    confthresh : float, optional
        Confidence threshold under which a prediction will be scrapped
        altogether. The default is 0.6.
    iouthresh : float, optional
        An intersection-over-union ratio over which two boxes will be
        considered as overlapping. The default is 0.5.

    Returns
    -------
    boxes: ArrayLike
        An array of N' samples in [x1, y1, x2, y2] format with the accepted
        boxes.
    confidence: ArrayLike
        An array of N' samples with each accepted box's confidence.
    labels: ArrayLike
        An array of N' samples with each accepted box's label.
    """
    out_box = []
    out_conf = []
    out_labels = []

    good = confidence >= confthresh
    boxes = boxes[good]
    confidence = confidence[good]
    labels = labels[good]

    sorting = np.argsort(confidence)
    boxes = boxes[sorting]
    confidence = confidence[sorting]
    labels = labels[sorting]

    while len(boxes > 0):
        box = boxes[0]
        out_box.append(box)
        out_conf.append(confidence[0])
        out_labels.append(labels[0])

        non_collisions = np.nonzero(
            iou(box[None, :], boxes, iouthresh).squeeze(axis=0) == 0
        )[0]
        boxes = boxes[non_collisions]
        confidence = confidence[non_collisions]
        labels = labels[non_collisions]

    return np.asarray(out_box), np.asarray(out_conf), np.asarray(out_labels)


def avg_precision(gt, pred):
    """Compute the Average Precision of a set of predictions.

    Computes the Average Precision of a series of predictions against a ground
    truth. They should be aligned.

    Parameters
    ----------
    gt : ArrayLike
        Array with ground-truth labels.
    pred : ArrayLike
        Array with predicted labels. A -1 indicates a non-valid prediction
        which will be considered a false negative.

    Returns
    -------
    AP : Float
        Average Precision value.

    """
    cumulative_tp = np.cumsum(gt == pred)
    cumulative_fn = np.cumsum(pred == -1)
    cumulative_fp = np.arange(1, len(cumulative_tp) + 1) - (
        cumulative_fn + cumulative_tp
    )

    step_recall = cumulative_tp / (cumulative_tp + cumulative_fp)
    prec_denominator = cumulative_tp + cumulative_fn
    step_precision = np.divide(
        cumulative_tp.astype(np.float),
        prec_denominator.astype(np.float),
        out=np.zeros(cumulative_tp.shape, dtype=np.float),
        where=prec_denominator != 0,
    )

    mono_precision = np.flip(np.maximum.accumulate(np.flip(step_precision)))

    offset_step_recall = np.insert(step_recall, 0, 0)
    AP = np.sum((step_recall - offset_step_recall[:-1]) * mono_precision)

    return AP


def levenshtein(
    source: Union[ArrayLike, str],
    target: Union[ArrayLike, str],
) -> Tuple[float, ArrayLike]:
    """Compute the Levenshtein distance between two strings.

    Parameters
    ----------
    source: Union[ArrayLike, str]
        An input sequence in a numpy array or a string.
    target: Union[ArrayLike, str]
        The sequence to compare to as a numpy array or a string.

    Returns
    -------
    float
        Levenshtein distance between both strings.
    ArrayLike
        Dynamic programming matrix for the string comparison process.
    """
    matrix = []

    if len(target) == 0:
        return len(source), np.array([])

    # We call tuple() to force strings to be used as sequences
    if not (isinstance(source, np.ndarray)):
        source = np.array(tuple(source))
    if not (isinstance(target, np.ndarray)):
        target = np.array(tuple(target))

    previous_row = np.arange(target.size + 1)

    matrix.append(previous_row)
    for s in source:
        current_row = previous_row + 1
        current_row[1:] = np.minimum(
            current_row[1:], np.add(previous_row[:-1], target != s)
        )
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)

        previous_row = current_row
        matrix.append(previous_row)

    return previous_row[-1] / float(len(target)), np.array(matrix)


def edit_path(matrix: ArrayLike) -> List[Tuple[int, int, str]]:
    """Generate the edit sequence from a Levenshtein output matrix.

    Parameters
    ----------
    matrix: ArrayLike
        The output of a Levenshtein algorithm. Should be a matrix of shape N x M where
        N is the number of characters in the predicted sequence and M is the number of
        characters in the target sequence.

    Returns
    -------
    List[Tuple[int, int, str]]
        A list of edits. The first two integers of each tuple are the position in the
        source and target strings respectively and the last string denotes the type of
        operation to perform ([sub]stitution, [rem]oval, [ins]ertion)
    """
    rows, cols = matrix.shape

    ii = rows - 1
    jj = cols - 1

    edits = []

    while ii > 0 or jj > 0:
        current_val = matrix[ii, jj]

        north = matrix[ii - 1, jj] if ii > 0 else math.inf
        east = matrix[ii, jj - 1] if jj > 0 else math.inf
        diag = matrix[ii - 1, jj - 1] if ii > 0 and jj > 0 else math.inf

        minval = min(north, east, diag)

        if minval == diag:
            ii = ii - 1
            jj = jj - 1
            if minval != current_val:
                edits.append((ii, jj, "sub"))
        elif minval == north:
            ii = ii - 1
            jj = jj
            edits.append((ii, jj, "rem"))
        elif minval == east:
            ii = ii
            jj = jj - 1
            edits.append((ii, jj, "ins"))
    edits.reverse()
    return edits


def edit_coords(
    coords: ArrayLike,
    edits: List[Tuple[int, int, str]],
    gt_len: int,
) -> ArrayLike:
    """Modify coordinates according to the string edit path.

    Parameters
    ----------
    coords: ArrayLike
        An array of predicted coordinates of shape N x 2.
    edits: List[Tuple[int, int, str]]
        A sequence of edits from the ```edit_path``` function. The first two integers of
        each tuple are the position in the source and target strings respectively and
        the last string denotes the type of operation to perform ([sub]stitution,
        [rem]oval, [ins]ertion).
    gt_len: int
        Length of the ground truth sequence.

    Returns
    -------
    ArrayLike
        Input coordinates modified to account for insertions and removals.
    """
    output = np.full((gt_len, 2), -1, dtype=int)

    start_pred = 0
    start_out = 0

    for pd_ind, gt_ind, edit in edits:
        if edit == "sub":
            continue

        segment = coords[start_pred:pd_ind]
        output[start_out : start_out + len(segment)] = segment
        start_out += len(segment)
        start_pred = pd_ind

        if edit == "rem":
            start_pred += 1
        if edit == "ins":
            start_out += 1

    segment = coords[start_pred : len(coords)]
    output[start_out : start_out + len(segment)]
    gaps = find_gaps(output)
    output = remove_gaps(output, gaps)

    return output


def find_gaps(output: ArrayLike) -> List[Tuple[int, int]]:
    """Find contiguous sets of invalid coordinates.

    Parameters
    ----------
    output: ArrayLike
        Coordinate prediction in N x 2 format.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples representing the starting and ending indices of an invalid
        block.
    """
    invalid = np.all(output == -1, axis=1)
    gaps = []

    start = -1
    last = False
    for index, curr_inv in enumerate(invalid):
        if curr_inv:
            if not last:
                start = index
                last = True
        else:
            if last:
                gaps.append((start, index))
                last = False

    if last:
        gaps.append((start, len(invalid)))

    return gaps


def remove_gaps(output: ArrayLike, gaps: List[Tuple[int, int]]) -> ArrayLike:
    """Replace invalid coordinates with plausible proposals.

    Parameters
    ----------
    output: ArrayLike
        Coordinate prediction in N x 2 format.
    gaps: List[Tuple[int, int]]
        A list of tuples representing the starting and ending indices of an invalid
        block.

    Returns
    -------
    ArrayLike
        The modified input coordinates.
    """
    invalid = np.all(output == -1, axis=1)
    median_size = output[np.logical_not(invalid)]
    median_size = median_size[:, 1] - median_size[:, 0]

    if len(median_size):
        median_size = np.median(median_size)
    else:
        median_size = 50

    for gap in gaps:
        start_index, end_index = gap
        nblocks = end_index - start_index
        start_coord = output[start_index - 1, 1] if start_index > 0 else 0
        end_coord = (
            output[end_index, 0]
            if end_index < len(invalid)
            else start_coord + ((end_index - start_index) * median_size)
        )
        block_size = (end_coord - start_coord) // nblocks
        for block in range(nblocks):
            output[start_index + block, 0] = start_coord + (block * block_size)
            output[start_index + block, 1] = start_coord + ((block + 1) * block_size)

    return output


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
