"""
Shared exception hierarchy for Clima-Core.
"""

class ClimaCoreError(Exception):
    """Base exception for all Clima-Core errors."""
    pass


class ConfigurationError(ClimaCoreError):
    """Raised when configuration is invalid or missing."""
    pass


class DataValidationError(ClimaCoreError):
    """Raised when data fails validation or schema expectations."""
    pass


class LeakageError(ClimaCoreError):
    """Raised when future-data leakage or target leakage is detected."""
    pass


class AlignmentError(ClimaCoreError):
    """Raised when spatial or temporal alignment between forecast and verification fails."""
    pass


class ModelTrainingError(ClimaCoreError):
    """Raised when training fails or diverges."""
    pass


class ProvenanceError(ClimaCoreError):
    """Raised when provenance, checksums, or artifact manifests fail verification."""
    pass
