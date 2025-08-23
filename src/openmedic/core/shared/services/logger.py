import logging
import logging.config
import os
from dataclasses import dataclass
from typing import Optional


_LOG_FILE_PATH: Optional[str] = None
_INITIALIZED: bool = False
# Track whether an in-place progress line is currently shown on console
_PROGRESS_ACTIVE: bool = False
_PROGRESS_LEN: int = 0


@dataclass
class LoggerOptions:
    """Options to configure experiment logging.

    Attributes:
    -----------
        filename: str
            Log file name to create inside the experiment directory.
        level: int
            Root logger level (e.g., logging.INFO).
        enable_console: bool
            Whether to attach a console stream handler.
        enable_color: bool
            Whether to colorize console output.
        use_rotation: bool
            Use size-based rotation if True; otherwise a single file handler is used.
        max_bytes: int
            Maximum bytes per log file before rotation when use_rotation=True.
        backup_count: int
            Number of rotated backup files to keep when use_rotation=True.
    """

    filename: str = "training.log"
    level: int = logging.INFO
    enable_console: bool = True
    enable_color: bool = True
    use_rotation: bool = False
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 3


class _ColorFormatter(logging.Formatter):
    """ANSI color formatter for console logs with single-line output.

    - INFO: green
    - WARNING: yellow
    - ERROR/CRITICAL: red
    - DEBUG: cyan (fallback)
    """

    _RESET: str = "\x1b[0m"
    _BOLD: str = "\x1b[1m"
    _CYAN: str = "\x1b[36m"
    _GREEN: str = "\x1b[32m"
    _YELLOW: str = "\x1b[33m"
    _RED: str = "\x1b[31m"

    def __init__(self, fmt: str):
        super().__init__(fmt=fmt)

    def format(self, record: logging.LogRecord) -> str:
        # Build a single-line message and color ONLY the level label
        raw_message: str = record.getMessage()
        message_one_line: str = " | ".join(part.strip() for part in raw_message.splitlines())
        # Bold any bracketed module tags like [OpenMedicManager][plan_train]
        try:
            import re  # Local import to avoid global dependency

            def _bold_brackets(m: "re.Match[str]") -> str:
                return f"{self._BOLD}{m.group(1)}{self._RESET}"

            message_one_line = re.sub(r"(\[[^\]]+\])", _bold_brackets, message_one_line)
        except Exception:
            pass
        level_label: str = record.levelname
        if record.levelno >= logging.CRITICAL:
            level_colored = f"{self._RED}{self._BOLD}{level_label}{self._RESET}"
        elif record.levelno >= logging.ERROR:
            level_colored = f"{self._RED}{level_label}{self._RESET}"
        elif record.levelno >= logging.WARNING:
            level_colored = f"{self._YELLOW}{level_label}{self._RESET}"
        elif record.levelno >= logging.INFO:
            level_colored = f"{self._GREEN}{level_label}{self._RESET}"
        else:
            level_colored = f"{self._CYAN}{level_label}{self._RESET}"
        # Keep logger name and message uncolored (default terminal color)
        return f"{level_colored}:{record.name}:{message_one_line}"


def _compute_experiment_dir() -> str:
    """Compute the current experiment directory path.

    This function lazily imports pipeline runtime state to avoid circular imports
    and returns the absolute path to the experiment directory under `.openmedic`.
    """
    from openmedic.core.shared.services.plans.management import (
        OpenMedicOSEnv,
        OpenMedicPipelineResult,
    )

    experiment_dir: str = os.path.join(
        OpenMedicOSEnv.home,
        OpenMedicPipelineResult.get_current_experiment(),
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def _configure_console(root_logger: logging.Logger, *, level: int, enable_color: bool) -> None:
    """Attach or update a console stream handler with optional colors."""
    fmt: str = "%(levelname)s:%(name)s:%(message)s"

    class ConsoleStreamHandler(logging.StreamHandler):
        """Stream handler that clears in-place progress lines before emitting logs.

        This prevents standard logs (e.g., checkpoint/save) from breaking the
        single-line training/eval progress line.
        """

        def emit(self, record: logging.LogRecord) -> None:
            try:
                global _PROGRESS_ACTIVE, _PROGRESS_LEN
                if _PROGRESS_ACTIVE and self.stream is not None:
                    try:
                        self.stream.write("\r" + (" " * max(_PROGRESS_LEN, 0)) + "\r")
                        self.flush()
                    except Exception:
                        pass
                    _PROGRESS_ACTIVE = False
                    _PROGRESS_LEN = 0
            except Exception:
                # Defensive: proceed with normal emit
                pass
            super().emit(record)
    has_stream_handler: bool = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)
            handler.setFormatter(_ColorFormatter(fmt) if enable_color else logging.Formatter(fmt))
            has_stream_handler = True
    if not has_stream_handler:
        stream_handler = ConsoleStreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(_ColorFormatter(fmt) if enable_color else logging.Formatter(fmt))
        root_logger.addHandler(stream_handler)


def _configure_file_handler(root_logger: logging.Logger, *, log_path: str, level: int, use_rotation: bool, max_bytes: int, backup_count: int) -> None:
    """Attach a file-based handler, avoiding duplicates pointing to the same file."""
    existing_same_file: bool = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if os.path.abspath(getattr(handler, "baseFilename", "")) == os.path.abspath(log_path):
                    existing_same_file = True
                    break
            except Exception:
                pass
    if existing_same_file:
        return

    if use_rotation:
        from logging.handlers import RotatingFileHandler

        file_handler: logging.Handler = RotatingFileHandler(
            log_path, mode="a", maxBytes=max_bytes, backupCount=backup_count
        )
    else:
        file_handler = logging.FileHandler(log_path, mode="w")

    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)


def setup_experiment_logger(*, log_filename: str = "training.log", options: Optional[LoggerOptions] = None, config_path: Optional[str] = None) -> str:
    """Configure root logger (file + console) for current experiment; idempotent.

    Input:
    ------
        log_filename: str
            The log file name to be created inside the experiment directory.
        options: Optional[LoggerOptions]
            Programmatic options for rotation, color, level, and console.
        config_path: Optional[str]
            Optional path to a YAML logging configuration. When provided and file
            exists, it will be loaded via logging.config.dictConfig. The handler
            file paths will be rewritten to the computed experiment directory if
            they are relative paths.

    Output:
    -------
        str - Absolute path to the created log file.

    Notes:
    ------
        This function is idempotent; subsequent calls will return the same path
        without re-adding handlers.
    """
    global _INITIALIZED, _LOG_FILE_PATH

    if _INITIALIZED and _LOG_FILE_PATH:
        return _LOG_FILE_PATH

    opts: LoggerOptions = options or LoggerOptions(filename=log_filename)
    experiment_dir: str = _compute_experiment_dir()
    log_path: str = os.path.join(experiment_dir, log_filename or opts.filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(opts.level)

    # Determine default YAML config path if not provided
    if not config_path:
        # services/logger.py -> shared -> core -> openmedic -> template/logging_cfg.yml
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "template", "logging_cfg.yml")
        )

    if config_path and os.path.isfile(config_path):
        try:
            import yaml

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            # Rewrite any handler filename fields to live under experiment_dir
            handlers = (cfg or {}).get("handlers", {})
            for h in handlers.values():
                filename = h.get("filename")
                if filename:
                    if not os.path.isabs(filename):
                        h["filename"] = os.path.join(experiment_dir, filename)
                    else:
                        # Keep absolute file paths as-is
                        pass
            logging.config.dictConfig(cfg)
            # Try to infer log file path from configured handlers
            _LOG_FILE_PATH = None
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.FileHandler):
                    try:
                        _LOG_FILE_PATH = os.path.abspath(getattr(handler, "baseFilename", log_path))
                        break
                    except Exception:
                        pass
            if _LOG_FILE_PATH is None:
                _LOG_FILE_PATH = log_path
        except Exception:
            # Fallback to programmatic setup when config load fails
            _configure_file_handler(
                root_logger,
                log_path=log_path,
                level=opts.level,
                use_rotation=opts.use_rotation,
                max_bytes=opts.max_bytes,
                backup_count=opts.backup_count,
            )
    else:
        _configure_file_handler(
            root_logger,
            log_path=log_path,
            level=opts.level,
            use_rotation=opts.use_rotation,
            max_bytes=opts.max_bytes,
            backup_count=opts.backup_count,
        )

    if opts.enable_console:
        _configure_console(root_logger, level=opts.level, enable_color=opts.enable_color)

    _INITIALIZED = True
    _LOG_FILE_PATH = _LOG_FILE_PATH or log_path
    logging.info(f"[ExperimentLogger]: Logging to {_LOG_FILE_PATH}")
    return _LOG_FILE_PATH


def get_experiment_log_path() -> Optional[str]:
    """Return the absolute path to the log file if initialized."""
    return _LOG_FILE_PATH


class TrainingConsole:
    """Pretty, minimal console output for training with optional colors.

    This console prints a compact header and per-epoch lines using metrics
    available after each epoch. It does not write to files; file logging is
    handled by setup_experiment_logger.
    """

    _RESET: str = "\x1b[0m"
    _BOLD: str = "\x1b[1m"
    _CYAN: str = "\x1b[36m"
    _GREEN: str = "\x1b[32m"
    _YELLOW: str = "\x1b[33m"
    _MAGENTA: str = "\x1b[35m"
    _last_line_len: int = 0

    def _supports_color(self) -> bool:
        """Check if stdout likely supports ANSI colors."""
        try:
            import sys

            return sys.stdout.isatty()
        except Exception:
            return False

    def _c(self, text: str, color: str) -> str:
        """Colorize text if supported; otherwise return plain text."""
        if self._supports_color():
            return f"{color}{text}{self._RESET}"
        return text

    def _get_gpu_mem(self) -> str:
        """Return current GPU reserved memory as string like '11.9G'."""
        try:
            import torch

            if torch.cuda.is_available():
                mem_bytes: int = torch.cuda.memory_reserved(0)
                mem_gb: float = mem_bytes / 1e9
                return f"{mem_gb:.1f}G"
        except Exception:
            pass
        return "-"

    @staticmethod
    def _visible_len(text: str) -> int:
        """Return length without ANSI escape codes for proper padding."""
        try:
            import re

            return len(re.sub(r"\x1b\[[0-9;]*m", "", text))
        except Exception:
            return len(text)

    def _print_progress_line(self, line: str) -> None:
        """Render a progress line in-place and clear leftovers from prior longer lines."""
        visible_len: int = self._visible_len(line)
        pad_len: int = max(self._last_line_len - visible_len, 0)
        print(f"\r{line}{' ' * pad_len}", end="", flush=True)
        self._last_line_len = visible_len
        # Mark progress as active so console handler can clear it before logs
        try:
            global _PROGRESS_ACTIVE, _PROGRESS_LEN
            _PROGRESS_ACTIVE = True
            _PROGRESS_LEN = visible_len + pad_len
        except Exception:
            pass

    def print_header(self) -> None:
        """Print a two-line header similar to the demo in testlog.py."""
        line1: str = f"{'Epoch':>10} {'GPU_mem':>10} {'train_loss':>12} {'eval_loss':>12} {'train_acc':>12} {'eval_acc':>12}"

        print(self._c(self._BOLD + line1, self._CYAN))

    def print_epoch(
        self,
        *,
        epoch_idx: int,
        num_epochs: int,
        train_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        train_metric: Optional[float] = None,
        eval_metric: Optional[float] = None,
    ) -> None:
        """Print one epoch's summary as two aligned lines with colors.

        Input:
        ------
            epoch_idx: int - Current epoch (1-based)
            num_epochs: int - Total epochs
            train_loss/eval_loss: Optional[float] - Mean loss values
            train_metric/eval_metric: Optional[float] - Mean accuracy/metric values
        """
        # Clear any in-place progress line before printing final epoch summary
        try:
            print("\r" + (" " * 160) + "\r", end="", flush=True)
            # Reset progress line length because we cleared the line
            self._last_line_len = 0
        except Exception:
            pass
        # Progress is no longer active after printing the epoch summary line
        try:
            global _PROGRESS_ACTIVE, _PROGRESS_LEN
            _PROGRESS_ACTIVE = False
            _PROGRESS_LEN = 0
        except Exception:
            pass
        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        epoch_cell: str = self._c(f"{epoch_str:>10}", self._MAGENTA)
        gpu_mem: str = self._get_gpu_mem()

        train_loss_s: str = "-" if train_loss is None else f"{train_loss:.4f}"
        eval_loss_s: str = "-" if eval_loss is None else f"{eval_loss:.4f}"
        train_metric_s: str = "-" if train_metric is None else f"{train_metric:.4f}"
        eval_metric_s: str = "-" if eval_metric is None else f"{eval_metric:.4f}"

        epoch_line: str = (
            f"{epoch_cell} "
            f"{gpu_mem:>10} "
            f"{self._c(f'{train_loss_s:>12}', self._YELLOW)} "
            f"{self._c(f'{eval_loss_s:>12}', self._YELLOW)} "
            f"{self._c(f'{train_metric_s:>12}', self._GREEN)} "
            f"{self._c(f'{eval_metric_s:>12}', self._GREEN)}"
        )
        print(epoch_line)

    def print_step_progress(
        self,
        *,
        epoch_idx: int,
        num_epochs: int,
        step_idx: int,
        total_steps: int,
        train_loss_running: Optional[float] = None,
        train_metric_running: Optional[float] = None,
    ) -> None:
        """Print a single updating progress line for training steps.

        The line overwrites itself using a carriage return and no newline, so it
        continuously updates until the epoch completes.
        """
        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        gpu_mem: str = self._get_gpu_mem()
        tl_s: str = "-" if train_loss_running is None else f"{train_loss_running:.4f}"
        tm_s: str = "-" if train_metric_running is None else f"{train_metric_running:.4f}"
        eval_metric_placeholder: str = f"{'-':>12}"
        line: str = (
            f"{self._c(f'{epoch_str:>10}', self._MAGENTA)} "
            f"{gpu_mem:>10} "
            f"{self._c(f'{tl_s:>12}', self._YELLOW)} "
            f"{'-':>12} "
            f"{self._c(f'{tm_s:>12}', self._GREEN)} "
            f"{self._c(eval_metric_placeholder, self._GREEN)} "
            f"  [step {step_idx}/{total_steps}]"
        )
        self._print_progress_line(line)
        
    def print_eval_step_progress(
        self,
        *,
        epoch_idx: int,
        num_epochs: int,
        step_idx: int,
        total_steps: int,
        eval_loss_running: Optional[float] = None,
        eval_metric_running: Optional[float] = None,
    ) -> None:
        """Print a single updating progress line for evaluation steps.

        This prints in-place using a carriage return and no newline to keep the
        output as a single bottom line and avoid breaking the summary table.
        """
        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        gpu_mem: str = self._get_gpu_mem()
        el_s: str = "-" if eval_loss_running is None else f"{eval_loss_running:.4f}"
        em_s: str = "-" if eval_metric_running is None else f"{eval_metric_running:.4f}"
        line: str = (
            f"{self._c(f'{epoch_str:>10}', self._MAGENTA)} "
            f"{gpu_mem:>10} "
            f"{'-':>12} "
            f"{self._c(f'{el_s:>12}', self._YELLOW)} "
            f"{'-':>12} "
            f"{self._c(f'{em_s:>12}', self._GREEN)} "
            f"  [eval step {step_idx}/{total_steps}]"
        )
        self._print_progress_line(line)
        # When running standalone evaluation (usually num_epochs == 1),
        # print a newline at the end so subsequent logs don't attach.
        if step_idx >= total_steps and num_epochs == 1:
            print("\n")
            self._last_line_len = 0