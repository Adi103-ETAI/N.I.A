"""P1 Cron extensions — lifecycle guard, ReadWriteLock, pools, credential guard,
wake gate, origin tracking, toolset resolver, continuable threads, blueprint
catalog, suggestions engine.

This package contains the P1 cron subsystem extensions. The existing
cron implementation lives in ``niaharness/services/cron.py`` +
``niaharness/services/cron_scheduler.py`` + ``niaharness/services/cron_delivery.py``.
This package adds the missing features from AUDIT.md without modifying
the existing files' core logic — it provides new modules that the
existing code calls into.
"""

from niaharness.cron.blueprint_catalog import (
    AutomationBlueprint,
    BlueprintFillError,
    BlueprintSlot,
    fill_blueprint,
    get_blueprint,
    list_blueprints,
)
from niaharness.cron.continuable_threads import (
    deliver_to_thread_or_mirror,
    open_continuable_cron_thread,
    seed_cron_thread_session,
)
from niaharness.cron.credential_guard import (
    guard_job_credential_exfil,
    validate_cron_base_url,
)
from niaharness.cron.lifecycle_guard import (
    GatewayLifecycleBlocked,
    check_gateway_lifecycle,
    contains_gateway_lifecycle_command,
)
from niaharness.cron.origin import (
    cron_job_origin_log_suffix,
    cron_mirror_delivery_enabled,
    maybe_mirror_cron_delivery,
    resolve_origin,
    target_matches_origin,
)
from niaharness.cron.pools import (
    get_parallel_pool,
    get_sequential_pool,
    shutdown_pools,
    submit_parallel,
    submit_sequential,
)
from niaharness.cron.readwrite_lock import ReadWriteLock, terminal_cwd_lock
from niaharness.cron.suggestions import (
    accept_suggestion,
    add_suggestion,
    clear_all,
    clear_resolved,
    dismiss_suggestion,
    get_suggestion,
    list_accepted,
    list_dismissed,
    list_pending,
    load_suggestions,
)
from niaharness.cron.toolset_resolver import (
    merge_mcp_into_per_job_toolsets,
    resolve_cron_disabled_toolsets,
    resolve_cron_enabled_toolsets,
    resolve_cron_toolsets,
)
from niaharness.cron.wake_gate import build_wake_gate_output, parse_wake_gate

__all__ = [
    "AutomationBlueprint",
    "BlueprintFillError",
    "BlueprintSlot",
    "GatewayLifecycleBlocked",
    "ReadWriteLock",
    "accept_suggestion",
    "add_suggestion",
    "build_wake_gate_output",
    "check_gateway_lifecycle",
    "clear_all",
    "clear_resolved",
    "contains_gateway_lifecycle_command",
    "cron_job_origin_log_suffix",
    "cron_mirror_delivery_enabled",
    "deliver_to_thread_or_mirror",
    "dismiss_suggestion",
    "fill_blueprint",
    "get_blueprint",
    "get_parallel_pool",
    "get_sequential_pool",
    "get_suggestion",
    "guard_job_credential_exfil",
    "list_accepted",
    "list_blueprints",
    "list_dismissed",
    "list_pending",
    "load_suggestions",
    "maybe_mirror_cron_delivery",
    "merge_mcp_into_per_job_toolsets",
    "open_continuable_cron_thread",
    "parse_wake_gate",
    "resolve_cron_disabled_toolsets",
    "resolve_cron_enabled_toolsets",
    "resolve_cron_toolsets",
    "resolve_origin",
    "seed_cron_thread_session",
    "shutdown_pools",
    "submit_parallel",
    "submit_sequential",
    "target_matches_origin",
    "terminal_cwd_lock",
    "validate_cron_base_url",
]
