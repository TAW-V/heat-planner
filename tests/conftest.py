"""pytest setup: make `custom_components` importable from the repository root."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
