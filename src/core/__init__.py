"""Cross-project infrastructure: input coercion, error reporting, model base."""
from src.core.forecaster import Forecaster
from src.core.metrics import ErrorReport
from src.core.validation import as_series, requires_fit, validates_series

__all__ = ["Forecaster", "ErrorReport", "as_series", "requires_fit", "validates_series"]
