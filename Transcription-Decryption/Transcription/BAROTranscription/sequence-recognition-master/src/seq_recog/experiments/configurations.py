"""Configurations for experiments."""

from pydantic import BaseModel


class DirectoryConfig(BaseModel):
    """Basic Directory settings."""

    # Define where the results will be stored
    results_dir: str

    # Define the root path for the data. All other paths are to be stored in
    # relation to this one.
    base_data_dir: str


class DecryptDirectoryConfig(DirectoryConfig):
    """Directories for Decrypt elements."""

    training_file: str
    training_root: str
    validation_file: str
    validation_root: str
    test_file: str
    test_root: str

    vocab_data: str


class ComrefDirectoryConfig(DirectoryConfig):
    """Directories for COMREF databases."""

    splits_file: str
    vocab_data: str


class ProtoComrefDirectoryConfig(DirectoryConfig):
    """Directories for COMREF databases."""

    splits_file: str
    prm_vocab_data: str
    sec_vocab_data: str
