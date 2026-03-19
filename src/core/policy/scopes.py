from enum import Enum

class CapabilityScope(str, Enum):
    """
    Defines the rigid authorization levels for any tool in the N.I.A framework.
    All tools must declare exactly one of these scopes.
    """
    READ_ONLY   = "read_only"    # always auto-approved
    WRITE       = "write"        # requires plan approval
    EXECUTE     = "execute"      # requires plan approval
    NETWORK     = "network"      # requires plan approval
    AGENT_SPAWN = "agent_spawn"  # requires plan approval
    DESTRUCTIVE = "destructive"  # requires explicit flag + approval
