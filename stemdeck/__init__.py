"""StemDeck - stem-based remix and DJ tool (milestone 1: engine core)."""
import os

__version__ = "0.1.0"

# AMD's GPU library (MIOpen) prints a stream of warnings while it tunes itself
# the first time a model runs on a card, e.g. "File is unreadable ...fdb.txt"
# (AMD ships no pre-tuned database for Radeon cards and builds one on first
# use) and "IsEnoughWorkspace ... GemmFwdRest" (a candidate routine it tried
# and skipped). They are noise for normal use, so only errors are shown.
# To see everything again, set MIOPEN_LOG_LEVEL=4 before starting StemDeck.
# This must happen before PyTorch is imported, hence here.
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
