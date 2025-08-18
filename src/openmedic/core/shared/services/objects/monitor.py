from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging


class OpenMedicMonitorOpBase(ABC):
    @classmethod
    def get_name(cls):
        return cls.__name__

    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def initialize():
        pass

    @abstractmethod
    def execute():
        pass


class OpenMedicMonitorOpError(Exception):
    def __init__(self, message: str = ""):
        self.message: str = message
        super().__init__(self.message)


class OpenMedicMonitor:
    logger = None
    epoch_history: List[Dict] = []

    def __call__() -> List[object]:
        return [OpenMedicMonitor.logger] if OpenMedicMonitor.logger else []

    @classmethod
    def add_op(cls, op_name: str, op_class: OpenMedicMonitorOpBase):
        setattr(cls, op_name, op_class)

    @classmethod
    def get_op(cls, op_name: str) -> OpenMedicMonitorOpBase:
        try:
            return getattr(cls, op_name)
        except AttributeError:
            raise OpenMedicMonitorOpError(f"The {op_name} operation does not exist.")

    @classmethod
    def _ensure_logger(cls):
        if cls.logger is not None:
            return
        try:
            from openmedic.core.shared.services.objects.ops.monitors.tensorboard import TensorBoard
        except Exception as ex:
            raise OpenMedicMonitorOpError(f"Cannot import TensorBoard: {ex}")

        try:
            if hasattr(TensorBoard, "initialize"):
                cls.logger = TensorBoard.initialize(is_activate=True)
            else:
                cls.logger = TensorBoard(writer=None, is_activate=True)  # type: ignore
        except Exception as ex:
            raise OpenMedicMonitorOpError(f"Cannot initialize TensorBoard: {ex}")

    @classmethod
    def log_epoch(cls, result: Dict):
        if not isinstance(result, dict):
            logging.warning("[OpenMedicMonitor] result is not a dict; skip.")
            return

        try:
            cls._ensure_logger()
        except Exception as ex:
            logging.warning(f"[OpenMedicMonitor] Skip logging (no logger): {ex}")
            cls.epoch_history.append(dict(result))
            return

        numeric = {k: v for k, v in result.items() if isinstance(v, (int, float))}
        epoch = result.get("epoch", getattr(cls.logger, "epoch", 0))

        try:
            if hasattr(cls.logger, "_log_metrics"):
                cls.logger._log_metrics(numeric, epoch=epoch) 
        except Exception as ex:
            logging.warning(f"[OpenMedicMonitor] logger metrics failed: {ex}")

        try:
            if hasattr(cls.logger, "_save_epoch_result"):
                cls.logger._save_epoch_result(result) 
        except Exception:
            pass

        if "epoch" not in result:
            try:
                if hasattr(cls.logger, "_step_epoch"):
                    cls.logger._step_epoch() 
                else:
                    cls.logger.epoch = getattr(cls.logger, "epoch", 0) + 1 
            except Exception:
                pass

        cls.epoch_history.append(dict(result))

    @classmethod
    def close(cls):
        if cls.logger is None:
            return
        try:
            if hasattr(cls.logger, "close"):
                cls.logger.close()
            elif hasattr(cls.logger, "_close"):
                cls.logger._close()
        finally:
            cls.logger = None

    @classmethod
    def get_all_results(cls) -> List[Dict]:
        return list(cls.epoch_history)
