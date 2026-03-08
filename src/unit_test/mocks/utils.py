import shutil


def count_calls(f):
    """Decorator to test, how often a function was called.
    Adjusted from:
    https://stackoverflow.com/questions/21716940/is-there-a-way-to-track-the-number-of-times-a-function-is-called/21717084"""
    def count_and_call(*args, **kwargs):
        count_and_call.call_count += 1
        return f(*args, **kwargs)

    count_and_call.call_count = 0

    return count_and_call


def clean_experiment(exp):
    shutil.rmtree(exp.args.log_path, ignore_errors=False, onerror=None)
