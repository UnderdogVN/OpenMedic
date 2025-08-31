import logging
import logging.config
import os
from typing import Optional


class OpenMedicLogger:
    """Unified logger with console helpers and no nested classes."""

    _DEFAULT_FILENAME: str = "openmedic.log"

    def __init__(self) -> None:
        self._initialized: bool = False
        self._log_file_path: Optional[str] = None
        self._progress_active: bool = False
        self._progress_len: int = 0

    # ---------- Core setup ----------
    def _compute_experiment_dir(self) -> str:
        """Return the absolute experiment directory path and ensure existence."""
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

    def _configure_console(self, root_logger: logging.Logger, *, level: int, enable_color: bool) -> None:
        """Attach/update a console handler; use this instance as the single filter/formatter."""
        fmt: str = "%(message)s"
        has_stream_handler: bool = False
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
                handler.setFormatter(logging.Formatter(fmt))
                if self not in getattr(handler, "filters", []):
                    handler.addFilter(self)  # self.filter(record) will be invoked
                has_stream_handler = True
        if not has_stream_handler:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(fmt))
            handler.addFilter(self)
            root_logger.addHandler(handler)

        # Persist flag so self.filter() knows whether to colorize
        self._console_color_enabled: bool = bool(enable_color)

    # logging.Filter - used by console handler to format message and clear progress
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            raw = record.getMessage()
            one = " | ".join(part.strip() for part in raw.splitlines())
            if getattr(self, "_console_color_enabled", False):
                try:
                    import re
                    one = re.sub(r"(\[[^\]]+\])", lambda m: "\x1b[1m" + m.group(1) + "\x1b[0m", one)
                except Exception:
                    pass
            lvl = record.levelname
            if getattr(self, "_console_color_enabled", False):
                if record.levelno >= logging.CRITICAL:
                    lvl = f"\x1b[31m\x1b[1m{lvl}\x1b[0m"
                elif record.levelno >= logging.ERROR:
                    lvl = f"\x1b[31m{lvl}\x1b[0m"
                elif record.levelno >= logging.WARNING:
                    lvl = f"\x1b[33m{lvl}\x1b[0m"
                elif record.levelno >= logging.INFO:
                    lvl = f"\x1b[32m{lvl}\x1b[0m"
                else:
                    lvl = f"\x1b[36m{lvl}\x1b[0m"
            # Clear progress line before changing msg so the final output is clean
            try:
                handler_stream = getattr(record, "stream", None)
                if self._progress_active and handler_stream is not None:
                    try:
                        handler_stream.write("\r" + (" " * max(self._progress_len, 0)) + "\r"); handler_stream.flush()
                    except Exception:
                        pass
                    self._progress_active = False; self._progress_len = 0
            except Exception:
                pass
            record.msg = f"{lvl}:{record.name}:{one}"
        except Exception:
            pass
        return True

    def _configure_file_handler(
        self,
        root_logger: logging.Logger,
        *,
        log_path: str,
        level: int,
        use_rotation: bool,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        """Attach a file handler, avoiding duplicates to the same path."""
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

    def ensure_initialized(
        self,
        *,
        filename: Optional[str] = None,
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_color: bool = True,
        use_rotation: bool = False,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 3,
        config_path: Optional[str] = None,
    ) -> str:
        """Initialize logging once and return the active log file path."""
        if self._initialized and self._log_file_path:
            return self._log_file_path

        chosen_filename: str = filename or self._DEFAULT_FILENAME
        experiment_dir: str = self._compute_experiment_dir()
        log_path: str = os.path.join(experiment_dir, chosen_filename)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        if not config_path:
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "template", "logging_cfg.yml")
            )

        if config_path and os.path.isfile(config_path):
            try:
                import yaml

                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                handlers = (cfg or {}).get("handlers", {})
                for h in handlers.values():
                    filename_field = h.get("filename")
                    if filename_field:
                        if not os.path.isabs(filename_field):
                            h["filename"] = os.path.join(experiment_dir, filename_field)
                logging.config.dictConfig(cfg)
                inferred_path: Optional[str] = None
                for handler in logging.getLogger().handlers:
                    if isinstance(handler, logging.FileHandler):
                        try:
                            inferred_path = os.path.abspath(getattr(handler, "baseFilename", log_path))
                            break
                        except Exception:
                            pass
                self._log_file_path = inferred_path or log_path
            except Exception:
                self._configure_file_handler(root_logger, log_path=log_path, level=level, use_rotation=use_rotation, max_bytes=max_bytes, backup_count=backup_count)
                self._log_file_path = log_path
        else:
            self._configure_file_handler(root_logger, log_path=log_path, level=level, use_rotation=use_rotation, max_bytes=max_bytes, backup_count=backup_count)
            self._log_file_path = log_path

        if enable_console:
            self._configure_console(root_logger, level=level, enable_color=enable_color)

        self._initialized = True
        logging.info(f"[ExperimentLogger]: Logging to {self._log_file_path}")
        return self._log_file_path

    def init(
        self,
        *,
        mode: Optional[str] = None,
        filename: Optional[str] = None,
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_color: bool = True,
        use_rotation: bool = False,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 3,
        config_path: Optional[str] = None,
    ) -> str:
        """Initialize with mode-specific defaults if provided, idempotently."""
        mode_to_file = {
            "train": "training.log",
            "eval": "evaluation.log",
            "infer": "inference.log",
            "inference": "inference.log",
        }
        chosen_filename: Optional[str] = filename or (mode_to_file.get(mode) if mode else None)
        return self.ensure_initialized(
            filename=chosen_filename,
            level=level,
            enable_console=enable_console,
            enable_color=enable_color,
            use_rotation=use_rotation,
            max_bytes=max_bytes,
            backup_count=backup_count,
            config_path=config_path,
        )

    # ---------- Proxies to stdlib logging ----------
    def info(self, msg: str, *args, **kwargs) -> None:
        self.ensure_initialized()
        logging.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.ensure_initialized()
        logging.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.ensure_initialized()
        logging.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.ensure_initialized()
        logging.debug(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self.ensure_initialized()
        logging.critical(msg, *args, **kwargs)

    # ---------- Console helpers (methods, no nested classes) ----------
    def _supports_color(self) -> bool:
        try:
            import sys

            return sys.stdout.isatty()
        except Exception:
            return False

    def _c(self, text: str, color: str) -> str:
        if self._supports_color():
            return f"{color}{text}\x1b[0m"
        return text

    @staticmethod
    def _visible_len(text: str) -> int:
        try:
            import re

            return len(re.sub(r"\x1b\[[0-9;]*m", "", text))
        except Exception:
            return len(text)

    def _get_gpu_mem(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                mem_bytes: int = torch.cuda.memory_reserved(0)
                mem_gb: float = mem_bytes / 1e9
                return f"{mem_gb:.1f}G"
        except Exception:
            pass
        return "-"

    def _print_progress_line(self, line: str) -> None:
        visible_len: int = self._visible_len(line)
        pad_len: int = max(getattr(self, "_last_line_len", 0) - visible_len, 0)
        print(f"\r{line}{' ' * pad_len}", end="", flush=True)
        self._last_line_len = visible_len
        self._progress_active = True
        self._progress_len = visible_len + pad_len

    def header(self) -> None:
        CYAN = "\x1b[36m"
        BOLD = "\x1b[1m"
        line1: str = f"{'Epoch':>10} {'GPU_mem':>10} {'train_loss':>12} {'eval_loss':>12} {'train_acc':>12} {'eval_acc':>12}"
        print(self._c(BOLD + line1, CYAN))

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
        try:
            print("\r" + (" " * 160) + "\r", end="", flush=True)
            self._last_line_len = 0
        except Exception:
            pass
        self._progress_active = False
        self._progress_len = 0

        MAGENTA = "\x1b[35m"
        YELLOW = "\x1b[33m"
        GREEN = "\x1b[32m"

        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        epoch_cell: str = self._c(f"{epoch_str:>10}", MAGENTA)
        gpu_mem: str = self._get_gpu_mem()

        train_loss_s: str = "-" if train_loss is None else f"{train_loss:.4f}"
        eval_loss_s: str = "-" if eval_loss is None else f"{eval_loss:.4f}"
        train_metric_s: str = "-" if train_metric is None else f"{train_metric:.4f}"
        eval_metric_s: str = "-" if eval_metric is None else f"{eval_metric:.4f}"

        epoch_line: str = (
            f"{epoch_cell} "
            f"{gpu_mem:>10} "
            f"{self._c(f'{train_loss_s:>12}', YELLOW)} "
            f"{self._c(f'{eval_loss_s:>12}', YELLOW)} "
            f"{self._c(f'{train_metric_s:>12}', GREEN)} "
            f"{self._c(f'{eval_metric_s:>12}', GREEN)}"
        )
        print(epoch_line)

    def train_progress(
        self,
        *,
        epoch_idx: int,
        num_epochs: int,
        step_idx: int,
        total_steps: int,
        train_loss_running: Optional[float] = None,
        train_metric_running: Optional[float] = None,
    ) -> None:
        MAGENTA = "\x1b[35m"
        YELLOW = "\x1b[33m"
        GREEN = "\x1b[32m"
        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        gpu_mem: str = self._get_gpu_mem()
        tl_s: str = "-" if train_loss_running is None else f"{train_loss_running:.4f}"
        tm_s: str = "-" if train_metric_running is None else f"{train_metric_running:.4f}"
        eval_metric_placeholder: str = f"{'-':>12}"
        line: str = (
            f"{self._c(f'{epoch_str:>10}', MAGENTA)} "
            f"{gpu_mem:>10} "
            f"{self._c(f'{tl_s:>12}', YELLOW)} "
            f"{'-':>12} "
            f"{self._c(f'{tm_s:>12}', GREEN)} "
            f"{self._c(eval_metric_placeholder, GREEN)} "
            f"  [step {step_idx}/{total_steps}]"
        )
        self._print_progress_line(line)

    def eval_progress(
        self,
        *,
        epoch_idx: int,
        num_epochs: int,
        step_idx: int,
        total_steps: int,
        eval_loss_running: Optional[float] = None,
        eval_metric_running: Optional[float] = None,
    ) -> None:
        MAGENTA = "\x1b[35m"
        YELLOW = "\x1b[33m"
        GREEN = "\x1b[32m"
        epoch_str: str = f"{epoch_idx}/{num_epochs}"
        gpu_mem: str = self._get_gpu_mem()
        el_s: str = "-" if eval_loss_running is None else f"{eval_loss_running:.4f}"
        em_s: str = "-" if eval_metric_running is None else f"{eval_metric_running:.4f}"
        line: str = (
            f"{self._c(f'{epoch_str:>10}', MAGENTA)} "
            f"{gpu_mem:>10} "
            f"{'-':>12} "
            f"{self._c(f'{el_s:>12}', YELLOW)} "
            f"{'-':>12} "
            f"{self._c(f'{em_s:>12}', GREEN)} "
            f"  [eval step {step_idx}/{total_steps}]"
        )
        self._print_progress_line(line)
        if step_idx >= total_steps and num_epochs == 1:
            print("\n")
            self._last_line_len = 0

    def train(self, **kwargs) -> None:
        if {"step_idx", "total_steps"}.issubset(kwargs.keys()):
            self.train_progress(**kwargs)
        else:
            self.print_epoch(**kwargs)

    def eval(self, **kwargs) -> None:
        if {"step_idx", "total_steps"}.issubset(kwargs.keys()):
            self.eval_progress(**kwargs)
        else:
            self.print_epoch(**kwargs)

    def checkpoint(self, message: str) -> None:
        self.info(message)

    # ---------- Introspection ----------
    def get_log_path(self) -> Optional[str]:
        """Return the absolute path to the active log file if initialized."""
        return self._log_file_path

# Default module-level singleton instance
logger: OpenMedicLogger = OpenMedicLogger()