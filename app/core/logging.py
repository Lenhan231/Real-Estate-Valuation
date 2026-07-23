"""Production logging configuration."""
import logging
import sys
from typing import Optional
from datetime import datetime
import os

# Configure logging for production
def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging for production."""

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File handler (optional)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_file:
        root_logger.addHandler(file_handler)

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("supabase").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Initialize logger for the app
logger = logging.getLogger(__name__)


class ProductionLogger:
    """Production-ready logging helper."""

    @staticmethod
    def log_prediction(
        locality: str,
        property_type: str,
        predicted_price: float,
        confidence: float,
        duration_ms: float,
    ) -> None:
        """Log prediction request."""
        logger.info(
            f"PREDICTION | locality={locality} | type={property_type} | "
            f"price={predicted_price:.2e} | confidence={confidence:.1f}% | "
            f"duration={duration_ms:.0f}ms"
        )

    @staticmethod
    def log_feedback(
        predicted_price: float,
        actual_price: Optional[float],
        rating: str,
        status: str,
    ) -> None:
        """Log feedback submission."""
        logger.info(
            f"FEEDBACK | predicted={predicted_price:.2e} | "
            f"actual={actual_price or 'None'} | rating={rating} | status={status}"
        )

    @staticmethod
    def log_error(
        endpoint: str,
        error_type: str,
        error_msg: str,
        details: Optional[dict] = None,
    ) -> None:
        """Log error with context."""
        details_str = str(details) if details else ""
        logger.error(
            f"ERROR | endpoint={endpoint} | type={error_type} | "
            f"message={error_msg} | details={details_str}"
        )

    @staticmethod
    def log_retraining(
        status: str,
        samples_used: Optional[int] = None,
        mape: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log model retraining."""
        if status == "started":
            logger.info(f"RETRAIN | status=started")
        elif status == "completed":
            logger.info(
                f"RETRAIN | status=completed | samples={samples_used} | mape={mape:.2f}%"
            )
        else:
            logger.error(f"RETRAIN | status={status} | error={error}")

    @staticmethod
    def log_health_check(status: str, details: Optional[dict] = None) -> None:
        """Log health check result."""
        details_str = " | ".join(f"{k}={v}" for k, v in (details or {}).items())
        logger.info(f"HEALTH | status={status} | {details_str}")


# Setup logging when module is imported
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
