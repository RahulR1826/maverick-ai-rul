import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# On Windows, importing torch after scikit-learn/TensorFlow can fail with a DLL
# clash, so import it first when it is available.
try:
    import torch  # noqa: F401
except ImportError:
    pass
