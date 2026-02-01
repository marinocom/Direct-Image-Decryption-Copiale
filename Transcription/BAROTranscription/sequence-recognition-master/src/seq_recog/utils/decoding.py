"""Sequence to Coordinate decoding algorithms.

Various CTC and Seq2Seq decoding algorithms to produce the respective output
coordinates and/or sequences (depending on the nature of the algorithm).
"""

from __future__ import annotations

from itertools import groupby
from typing import List, Optional, NamedTuple, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .ops import seqiou, sequiou_multiple, levenshtein, edit_path, edit_coords


BLANK_CHARACTER: int = 0
NO_CHARACTER: int = -1
FIRST_ELEMENT: int = -1
MAX_WIDTH: int = 0


class Coordinate(NamedTuple):
    """Represents a 1D coordinate range for alignment."""

    x1: int
    x2: int


class Prediction:
    """Encapsulate a predicted sequence into a single object."""

    def __init__(
        self,
        coordinates: ArrayLike,
        confidences: ArrayLike,
        characters: ArrayLike,
    ) -> None:
        """Create a Prediction object.

        :param coordinates: A list of Coordinate objects that represent the starting and
        ending pixels of a given object in the line.
        :param confidences: Confidence values for each coordinate tuple.
        :param characters: The character represented in each prediction step.
        """
        self._coordinates = coordinates
        self._confidences = confidences
        self._characters = characters

    def __str__(self) -> str:
        """Return a string representation of the prediction object."""
        return f"Prediction with coordinates:\n{str(self._coordinates)}"

    def __repr__(self) -> str:
        """Return a string representation of the prediction object."""
        return str(self)

    def __iter__(
        self,
    ) -> Tuple[Coordinate | None, float, int]:
        """Iterate over Prediction elements.

        :returns: A Tuple containing a Coordinate object, a confidence value
        and the underlying character at each specific time step.
        """
        for pred, conf, char in zip(
            self._coordinates, self._confidences, self._characters
        ):
            yield pred, conf, char

    def compare(
        self,
        gt_coordinates: List[Coordinate] | ArrayLike,
    ) -> float:
        """Compute the mAP of the prediction against the ground truth.

        :param gt_coordinates: A list of coordinates or an array of shape n_points x 2
        where each point has the starting and ending coordinate of a character.
        :returns: The IoU Array of all bounding boxes in the prediction against all
        bounding boxes in the ground truth.
        """
        if isinstance(gt_coordinates, list):
            gt_coordinates = np.array(gt_coordinates)

        assert (
            len(gt_coordinates.shape) == 2 and gt_coordinates.shape[1] == 2
        ), "Invalid shape for coordinate comparison"

        pred_coordinates = self._coordinates

        return seqiou(pred_coordinates, gt_coordinates)

    def get_sequence(self) -> List[int]:
        """Return the underlying sequence of character in encoded format."""
        return self._characters

    def get_coords(self) -> ArrayLike:
        """Return the predicted pixel coordinates."""
        return self._coordinates

    def get_confidences(self) -> ArrayLike:
        """Return the confidences for all predictions."""
        return self._confidences

    @classmethod
    def from_ctc_decoding(
        cls,
        char_indices: ArrayLike,
        gt_sequence: ArrayLike,
        ctc_matrix: ArrayLike,
        column_size: float,
    ) -> Prediction:
        """Create Prediction from the output of a PrefixTree CTC decoding.

        :param char_indices: Array containing the class of each CTC column w.r.t. the
        input ground truth sequence. Essentially, the index of the character in the
        ground truth sequence.
        :param gt_sequence: Array containing the ground truth sequence to align to.
        :param ctc_matrix: (sequence length, batch size, class confidence) matrix
        produced by a model.
        :param column_size: Size in pixels of each column in the CTC model.
        """
        coordinates = []
        confidences = []
        characters = []

        current_char = None
        start_coordinate = -1
        partial_confidences = []

        for ind, char_index in enumerate(char_indices):
            if char_index == -1:
                continue
            if current_char is None:
                current_char = char_index
                start_coordinate = int(ind * column_size)
                partial_confidences.append(ctc_matrix[ind, gt_sequence[char_index]])
            else:
                # If a different char is found, save the previous one and reset
                if current_char != char_index:
                    characters.append(gt_sequence[current_char])
                    coordinates.append(
                        Coordinate(start_coordinate, int(ind * column_size))
                    )
                    confidences.append(np.exp(np.array(partial_confidences)).mean())

                    current_char = char_index
                    start_coordinate = int(ind * column_size)
                    partial_confidences = [ctc_matrix[ind, gt_sequence[char_index]]]

                # Otherwise keep accumulating confidences
                else:
                    partial_confidences.append(ctc_matrix[ind, gt_sequence[char_index]])
        if current_char is not None and start_coordinate != int(ind * column_size):
            characters.append(gt_sequence[current_char])
            coordinates.append(
                Coordinate(start_coordinate, int(len(ctc_matrix) * column_size))
            )
            confidences.append(np.array(partial_confidences).mean())

        coordinates = np.array(coordinates)
        confidences = np.array(confidences)
        characters = np.array(characters)

        return cls(coordinates, confidences, characters)

    @classmethod
    def from_detector(
        cls,
        pred_text: ArrayLike,
        gt_text: ArrayLike,
        coords: ArrayLike,
    ) -> Prediction:
        """Produce a prediction from the output of an OCR or detection system.

        Parameters
        ----------
        pred_text: ArrayLike
            The predicted sequence of tokens by the detector.
        gt_text: ArrayLike
            The ground truth text for the given image.
        coords: ArrayLike
            Predicted coordinates for the text tokens.

        Returns
        -------
        Prediction
            A Prediction object with due information.
        """
        dist, mat = levenshtein(pred_text, gt_text)
        edits = edit_path(mat)
        coords = edit_coords(coords, edits, len(gt_text))

        return cls(coords, np.ones(len(gt_text)), gt_text)

    @classmethod
    def from_text_coords(
        cls,
        coordinates: ArrayLike,
        confidences: ArrayLike,
        characters: ArrayLike,
    ) -> Prediction:
        """Create a prediction from an input file's data.

        Parameters
        ----------
        coordinates: ArrayLike
            A N x 2 array with start-end coordinates for every character.
        confidences: Optional[ArrayLike]
            The confidence value for each bounding box prediction.
        characters: ArrayLike
            The encoded ground truth sequence fed within an array.
        """
        raise NotImplementedError


class PredictionGroup:
    """Aggregate (Ensemble) multiple predictions."""

    def __init__(
        self,
        predictions: List[Prediction],
        gt_sequence: ArrayLike,
        names: Optional[List[str | None]] = None,
    ) -> None:
        """Create PredictionGroup object.
        
        Parameters
        ----------
        predictions: List[Prediction]
            A list of Prediction objects of the same image line. They must contain the
            same number of elements, as they are produced from the same ground truth
            sequence as reference.
        gt_sequence: ArrayLike
            The ground truth of the sequence being predicted.
        names: Optional[List[str | None]] = None
            A list of method names to keep track of results in an orderly fashion.
        """
        self._predictions = predictions
        self._gt_sequence = gt_sequence
        self._names = names or [None for _ in predictions]

    def __len__(self) -> int:
        """Return the number of predictions within the group."""
        return len(self._predictions)

    def add_prediction(
        self,
        prediction: Prediction,
        name: Optional[str] = None,
    ) -> None:
        """Append a prediction to an existing prediction group.

        Parameters
        ----------
        prediction: Prediction
            A prediction to be appended to the group.
        """
        self._predictions.append(prediction)
        self._names.append(name)

    def merge_prediction_groups(
        self,
        other: PredictionGroup,
    ) -> PredictionGroup:
        """Create a group with the contents of two sets of predictions.

        Parameters
        ----------
        other: PredictionGroup
            A second PredictionGroup to merge the data with.
        """
        assert len(self._gt_sequence) == len(
            other._gt_sequence
        ), "Predictions from different ground truth sequences."
        assert np.all(
            self._gt_sequence == other._gt_sequence
        ), "Predictions from different ground truth sequences."

        predictions = self._predictions + other._predictions
        names = self._names + other._names

        return PredictionGroup(predictions, self._gt_sequence, names)

    def find_anchors(
        self,
        iou_thresh: float,
    ) -> Prediction:
        """Compute high-consensus predictions.

        Given a set of predicted alignments, provide the main consensus elements among
        all methods. The idea is finding those predictions in which plenty of methods
        agree or whose properties make sense in the overall scheme of things (no huge
        characters, no missed elements, etc).

        Parameters
        ----------
        iou_thresh: float
            The IoU threshold at which to consider valid consensus.

        Returns
        -------
        Prediction
            A Prediction composed only of the anchor elements.
        """
        predictions = np.stack([x.get_coords() for x in self._predictions])
        iou_vector = sequiou_multiple(predictions)
        agreement = iou_vector >= iou_thresh

        newx1 = np.mean(predictions[:, :, 0], axis=0).astype(int)
        newx2 = np.mean(predictions[:, :, 1], axis=0).astype(int)
        newpreds = np.stack([newx1, newx2], axis=-1)
        nullboxes = np.full(newpreds.shape, -1)

        anchors = np.where(agreement[:, None], newpreds, nullboxes)

        return anchors


class PrefixNode:
    """Node within a prefix tree with the full parent nodes' transcription."""

    def __init__(
        self,
        character: int,
        char_index: int,
        parent: Optional[PrefixNode],
        confidence: float,
        column: int = -1,
    ) -> None:
        """Construct a PrefixNode.

        Parameters
        ----------
        character: int
            Character pertaining to the current node.
        char_index: int
            Index of the position of the character in the output string.
        parent: Optional[PrefixNode]
            Pointer to the parent node if it exists.
        confidence: float
            Cumulative log probability of the sequence after the inclusion of this node.
        column: int
            The column of the CTC matrix being processed.
        """
        self._character = character
        self._char_index = char_index
        self._parent = parent
        self._confidence = confidence
        self._column = column

    def __str__(self) -> str:
        return f"Prefix Node: index {self._char_index} conf {self._confidence}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def column(
        self,
    ) -> int:
        """Get the depth of this node (CTC Column being decoded)."""
        return self._column

    @property
    def character(self) -> int:
        """Produce the character associated to this node."""
        return self._character

    @character.setter
    def character(
        self,
        value: int,
    ) -> None:
        self._character = value

    @property
    def char_index(
        self,
    ) -> int:
        """Produce the character index associated to this node."""
        return self._char_index

    @char_index.setter
    def char_index(
        self,
        value: int,
    ) -> None:
        self._char_index = value

    @property
    def confidence(
        self,
    ) -> float:
        """Produce current log likelihood of the prefix tree."""
        return self._confidence

    @confidence.setter
    def confidence(
        self,
        value: float,
    ) -> float:
        self._confidence = value

    def expand(
        self,
        character: int,
        char_index: int,
        confidence: float,
    ) -> PrefixNode:
        """Add a PrefixNode child to the current node.

        :param character: Character pertaining to the current node.
        :param char_index: Index of the position of the character in the output
        string.
        :param confidence: Log confidence value of the current node addition.

        :returns: Produced node added as child to the current one.
        """
        cumulative_confidence = self._confidence + confidence

        node = PrefixNode(
            character, char_index, self, cumulative_confidence, self._column + 1
        )

        return node

    def produce_sequence(
        self,
    ) -> ArrayLike:
        """Produce the sequence associated to this element of the prefix tree.

        :returns: Character index sequence associated to this element of the
        prefix tree. If the confidence is required, simply call
        PrefixNode.confidence.
        """
        node = self
        decoding = [node._char_index]

        while node._parent is not None:
            node = node._parent
            decoding.append(node._char_index)

        decoding.reverse()
        return np.array(decoding)


# TODO Implement this but using a dynamic programming algorithm (I *think* it
# is possible, but I am not sure. For the time being, beam decoding should
# yield optimum results anyway).
class PrefixTree:
    """Compute the highest log likelihood decoding of a given GT sequence."""

    def __init__(
        self,
        output_sequence: ArrayLike,
        beam_width: int = MAX_WIDTH,
    ) -> None:
        """Create a PrefixTree for CTC decoding given the gt sequence.

        :param root_character: A PrefixNode object with the starting character
        to perform decoding.
        :param output_sequence: The gt sequence with character indices as
        elements of a Numpy array.
        :param beam_width: Number of decoded sequences to keep at a time step.
        """
        self._output_sequence = output_sequence

        first_char = PrefixNode(self._output_sequence[0], 0, None, 1.0, 0)
        self._beams: List[PrefixNode] = [first_char]
        self._ended: List[PrefixNode] = []
        self._beam_width = beam_width

    def complete(self) -> bool:
        """Check whether the set of beams is complete or not."""
        if len(self._ended):
            if not len(self._beams):
                return True
            elif self._ended[0].confidence > self._beams[0].confidence:
                return True
            else:
                return False
        else:
            if len(self._beams):
                return False
            else:
                return True

    def __str__(self) -> str:
        """Get a string representation of the object."""
        return f"Prefix tree with <={self._beam_width} beams:" + str(
            [str(x) for x in self._beams]
        )

    def __repr__(self) -> str:
        """Get a string representation of the object."""
        return str(self)

    def _filter_nodes(self, columns: int, nodelist: List[PrefixNode]) -> None:
        alive_nodes = []

        for node in nodelist:
            if (
                node.column == columns - 1
                and node.char_index == len(self._output_sequence) - 1
            ):
                self._ended.append(node)
            elif len(self._output_sequence) - node.char_index - 1 < (
                columns - node.column - 1
            ):
                alive_nodes.append(node)
        self._ended.sort(key=lambda x: x.confidence, reverse=True)
        alive_nodes.sort(key=lambda x: x.confidence, reverse=True)

        if self._beam_width > 0:
            alive_nodes = alive_nodes[: self._beam_width]

        self._beams = alive_nodes

    def decode(
        self,
        ctc_matrix: ArrayLike,
    ) -> ArrayLike:
        """Produce the most likely decoding of the gt sequence.

        :param ctc_matrix: A (SeqLength x NumClasses) numpy array with each
        character prediction's confidence at each time step.

        :returns: Highest likelihood character indices for the ground truth
        sequence.
        """
        while not self.complete():
            alive_nodes = []
            for node in self._beams:
                char_index = node.char_index
                character = node.character

                if not character == BLANK_CHARACTER:
                    # Add same character as current node into the expansion
                    alive_nodes.append(
                        node.expand(
                            self._output_sequence[char_index],
                            char_index,
                            ctc_matrix[node.column + 1][character],
                        )
                    )

                # Add blank character
                alive_nodes.append(
                    node.expand(
                        BLANK_CHARACTER,
                        char_index,
                        ctc_matrix[node.column + 1][BLANK_CHARACTER],
                    )
                )

                if char_index < len(self._output_sequence) - 1:
                    # Add next character
                    alive_nodes.append(
                        node.expand(
                            self._output_sequence[char_index + 1],
                            char_index + 1,
                            ctc_matrix[node.column + 1][
                                self._output_sequence[char_index + 1]
                            ],
                        )
                    )

            self._filter_nodes(ctc_matrix.shape[0], alive_nodes)

        return self._ended[0].produce_sequence()


def decode_ctc(
    ctc_mat: ArrayLike,
    out_seq: ArrayLike,
    column_size: ArrayLike,
    beam_width: int = MAX_WIDTH,
) -> List[Prediction]:
    """Produce the most likely decoding of out_seq.

    Iterate over all possible transcriptions of each batch in a ctc model
    and produce the most likely decoding of the ground truth sequence.

    :param ctc_mat: (sequence length, batch size, class confidence) matrix
    produced by a model.
    :param out_seq: (batch size, sequence length) list with number arrays as
    elements in the sequence (zero is strictly reserved for the blank symbol).
    """
    outputs = []
    ctc_mat = ctc_mat.transpose((1, 0, 2))
    batch_size, seqlen, classes = ctc_mat.shape

    for mat, transcript, csize in zip(ctc_mat, out_seq, column_size):
        tree = PrefixTree(transcript, beam_width)
        decoding = tree.decode(mat)
        prediction = Prediction.from_ctc_decoding(
            decoding,
            transcript,
            mat,
            csize,
        )
        outputs.append(prediction)
    return outputs


def decode_ctc_greedy(ctc_matrix: ArrayLike) -> List[ArrayLike]:
    """Decode a CTC output matrix greedily and produce a sequence list.

    :param ctc_matrix: An output matrix from a CTC-like model. Should have
    shape S x B x C, where S is the sequence length, B is the batch size and C
    is the number of classes.
    :returns: A list containing the decoded sequence.
    """
    ctc_matrix = ctc_matrix.transpose((1, 0, 2))
    output = []
    for sample in ctc_matrix:
        maximum = sample.argmax(axis=-1)
        output.append(np.array([k for k, g in groupby(maximum) if k != 0]))
    return output
