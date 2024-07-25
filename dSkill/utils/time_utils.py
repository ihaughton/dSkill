import time
from functools import wraps


def time_function(func):
    """Decorator to time a function and print the execution time and result."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Function {func.__name__} executed in {elapsed_time:.2f} seconds")
        print(f"Result: {result}")
        return result

    return wrapper


def frequency_decorator(func):
    """Decorator to measure and report the frequency of function calls
    in Hz, based solely on the time since the last call."""
    last_call_time = [None]

    @wraps(func)
    def wrapper(*args, **kwargs):
        current_time = time.time()
        if last_call_time[0] is not None:
            time_difference = current_time - last_call_time[0]
            if time_difference != 0:
                current_frequency = 1 / time_difference
                print(
                    f"Frequency of calls to {func.__name__}: {current_frequency:.2f} Hz"
                )
            else:
                print(f"Function {func.__name__} called in rapid succession.")
        else:
            print(f"First call of {func.__name__}")

        last_call_time[0] = current_time
        result = func(*args, **kwargs)
        return result

    return wrapper
