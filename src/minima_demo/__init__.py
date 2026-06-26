"""Public, reproducible benchmark demo for Minima cost-aware LLM model routing.

The demo drives the public ``minima-cli`` SDK against the hosted ``api.minima.sh``
service and measures, per task: cost, latency, tokens, accuracy, and the *margin* to
the most-effective model — against the all-premium, cheapest, and oracle baselines —
while Minima learns from feedback within a single run (the learning curve).
"""

__version__ = "0.1.0"
