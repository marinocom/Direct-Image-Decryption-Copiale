"""Implementation of a progress bar enunciation."""


def progress(epoch, mode, loss):
    return f"Epoch {epoch} [{mode} l: {loss:.04}] -> "
