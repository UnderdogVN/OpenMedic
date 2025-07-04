import functools
import gc
import logging
import re
import time

import torch

import openmedic.core.shared.services.utils as utils


def camel_to_snake(name: str) -> str:
    """Converts CamelCase or camelCase to snake_case.
    Example: 'CamelCase' -> 'camel_case'
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return snake_case


def montior(input_function):
    """A function wrapper handles:
        - try/catch mechanism.
        - Returning geniric output.
        - Releasing memory at the end.

    Usage:
    ```
        import openmedic.core.shared.helper as helper
        @helper.montior
        def main_function(*args, **kwargs):
            pass
    ```
    """

    @functools.wraps(input_function)
    def try_cath_wrapper(*args, **kwargs):
        status: str = "success"
        error_msg: str = ""
        return_value: dict = {}
        try:
            start: float = time.perf_counter()
            result: any = input_function(*args, **kwargs)
            stop: float = time.perf_counter()
            duration: float = round(stop - start, 3)
            if isinstance(result, dict):
                return_value["runner"] = result["runner"]
                return_value["timestamp"] = result["timestamp"]
            elif result:
                return_value["result"] = result

            logging.info(
                f"[{montior.__name__}]: Completed progress with duration: {duration}s.",
            )
            return_value["duration"] = duration
        except Exception as e:
            status = "failure"
            error_msg = e
            logging.error(f"[try_catch][Exception]: {e}")
        except utils.BreakLoop as e:
            logging.warning(f"[try_catch][BreakLoop]: {e}")
        finally:
            return_value.update({"status": status, "error_msg": error_msg})
            gc.collect()
            torch.cuda.empty_cache()
            logging.info(f"[try_catch]: Released memory.")
            return return_value

    return try_cath_wrapper
