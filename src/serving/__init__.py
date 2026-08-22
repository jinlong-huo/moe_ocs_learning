"""
src.serving — multi-tenant vLLM (vllm-metal) serving capture.

Runs the real vLLM V1 engine on Apple Silicon (vllm-metal / MLX backend)
with several concurrent tenants arriving on a serving schedule, and
captures per-tenant routing traces + per-step contention data in the
canonical ``RoutingTrace`` format.
"""

from src.serving.schema import (
    MultiTenantSession,
    SessionMeta,
    StepRecord,
    TenantSummary,
)
from src.serving.capture import MultiTenantCapture, install_hooks, restore_hooks
from src.serving.workload import Workload, build_workload

__all__ = [
    "MultiTenantSession",
    "SessionMeta",
    "StepRecord",
    "TenantSummary",
    "MultiTenantCapture",
    "install_hooks",
    "restore_hooks",
    "Workload",
    "build_workload",
]
