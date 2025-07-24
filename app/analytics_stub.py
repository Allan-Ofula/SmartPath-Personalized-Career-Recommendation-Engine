# analytics_stub.py
# Stub functions to prevent ImportError during cloud deployment or when analytics.py is excluded

def log_usage(*args, **kwargs):
    pass  # This does nothing, just avoids crashes

def load_usage_data():
    return []  # Return empty list or DataFrame
