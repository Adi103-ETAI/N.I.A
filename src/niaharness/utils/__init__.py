"""niaharness utilities package.

This package provides common utility functions for the niaharness framework,
ported from OpenClaude TypeScript utilities to idiomatic Python.
"""

from .context import (
    COMPACT_MAX_OUTPUT_TOKENS,
    ESCALATED_MAX_TOKENS,
    CAPPED_DEFAULT_MAX_TOKENS,
    MAX_OUTPUT_TOKENS_DEFAULT,
    MAX_OUTPUT_TOKENS_UPPER_LIMIT,
    MODEL_CONTEXT_WINDOW_DEFAULT,
    OPENAI_FALLBACK_CONTEXT_WINDOW,
    ContextPercentages,
    ModelOutputTokens,
    calculate_context_percentages,
    get_context_window_for_model,
    get_max_thinking_tokens_for_model,
    get_model_max_output_tokens,
    has_1m_context,
    is_1m_context_disabled,
    model_supports_1m,
)
from .diff import (
    CONTEXT_LINES,
    DIFF_TIMEOUT_MS,
    DiffHunk,
    FileEdit,
    adjust_hunk_line_numbers,
    count_lines_changed,
    format_diff_for_display,
    get_patch_for_display,
    get_patch_from_contents,
)
from .file_read import (
    FileEncoding,
    FileMetadata,
    LineEnding,
    detect_encoding,
    detect_line_endings,
    has_binary_extension,
    is_binary_content,
    read_file,
    read_file_with_metadata,
)
from .git import (
    GitFileStatus,
    find_canonical_git_root,
    find_git_root,
    get_branch,
    get_changed_files,
    get_default_branch,
    get_file_status,
    get_head,
    get_is_clean,
    get_is_git,
    get_is_head_on_remote,
    get_remote_url,
    get_repo_remote_hash,
    has_unpushed_commits,
    normalize_git_remote_url,
    stash_to_clean_state,
)
from .glob import (
    glob_files,
    is_glob_pattern,
    match_files,
)
from .hash_utils import (
    hash_content,
    hash_file,
    hash_string,
    short_hash,
)
from .json_utils import (
    load_json_file,
    parse_json_strict,
    pretty_json,
    safe_parse_json,
    save_json_file,
    strip_bom,
    to_json,
)
from .path import (
    contains_path_traversal,
    expand_path,
    get_directory_for_path,
    normalize_path_for_config_key,
    sanitize_path,
    to_relative_path,
)
from .platform import (
    SUPPORTED_PLATFORMS,
    LinuxDistroInfo,
    detect_vcs,
    get_linux_distro_info,
    get_platform,
    get_wsl_version,
)
from .process import (
    ProcessResult,
    find_executable,
    is_process_running,
    kill_process_tree,
    run_command,
    run_command_sync,
)
from .ripgrep import (
    RipgrepConfig,
    RipgrepTimeoutError,
    RipgrepUnavailableError,
    count_files_rounded_rg,
    get_ripgrep_install_hint,
    get_ripgrep_status,
    rip_grep,
    rip_grep_stream,
    resolve_ripgrep_config,
)
from .shell_quote import (
    escape_shell_special,
    shell_join,
    shell_quote,
    shell_split,
)
from .sleep import (
    sleep,
    with_timeout,
)
from .token_budget import (
    find_token_budget_positions,
    get_budget_continuation_message,
    parse_token_budget,
)
from .uuid import (
    create_agent_id,
    generate_uuid,
    validate_uuid,
)
from .validation import (
    assert_function,
    assert_in_range,
    assert_non_empty_string,
    assert_object,
    assert_positive_int,
    validate_array_of,
)
from .version import (
    NIAHARNESS_RELEASES_URL,
    PUBLIC_BUILD_VERSION,
    get_public_build_version,
    get_release_tag_url,
    normalize_public_version,
)

__all__ = [
    # context
    "MODEL_CONTEXT_WINDOW_DEFAULT",
    "OPENAI_FALLBACK_CONTEXT_WINDOW",
    "COMPACT_MAX_OUTPUT_TOKENS",
    "MAX_OUTPUT_TOKENS_DEFAULT",
    "MAX_OUTPUT_TOKENS_UPPER_LIMIT",
    "CAPPED_DEFAULT_MAX_TOKENS",
    "ESCALATED_MAX_TOKENS",
    "ContextPercentages",
    "ModelOutputTokens",
    "is_1m_context_disabled",
    "has_1m_context",
    "model_supports_1m",
    "get_context_window_for_model",
    "calculate_context_percentages",
    "get_model_max_output_tokens",
    "get_max_thinking_tokens_for_model",
    # diff
    "CONTEXT_LINES",
    "DIFF_TIMEOUT_MS",
    "DiffHunk",
    "FileEdit",
    "adjust_hunk_line_numbers",
    "count_lines_changed",
    "get_patch_from_contents",
    "get_patch_for_display",
    "format_diff_for_display",
    # file_read
    "LineEnding",
    "FileEncoding",
    "FileMetadata",
    "detect_encoding",
    "detect_line_endings",
    "read_file_with_metadata",
    "read_file",
    "is_binary_content",
    "has_binary_extension",
    # git
    "find_git_root",
    "find_canonical_git_root",
    "get_is_git",
    "get_head",
    "get_branch",
    "get_default_branch",
    "get_remote_url",
    "get_is_head_on_remote",
    "has_unpushed_commits",
    "get_is_clean",
    "get_changed_files",
    "GitFileStatus",
    "get_file_status",
    "normalize_git_remote_url",
    "get_repo_remote_hash",
    "stash_to_clean_state",
    # glob
    "glob_files",
    "match_files",
    "is_glob_pattern",
    # hash_utils
    "hash_content",
    "hash_file",
    "hash_string",
    "short_hash",
    # json_utils
    "strip_bom",
    "safe_parse_json",
    "parse_json_strict",
    "to_json",
    "pretty_json",
    "load_json_file",
    "save_json_file",
    # path
    "expand_path",
    "to_relative_path",
    "get_directory_for_path",
    "contains_path_traversal",
    "normalize_path_for_config_key",
    "sanitize_path",
    # platform
    "SUPPORTED_PLATFORMS",
    "LinuxDistroInfo",
    "get_platform",
    "get_wsl_version",
    "get_linux_distro_info",
    "detect_vcs",
    # process
    "ProcessResult",
    "run_command",
    "run_command_sync",
    "find_executable",
    "kill_process_tree",
    "is_process_running",
    # ripgrep
    "RipgrepConfig",
    "RipgrepTimeoutError",
    "RipgrepUnavailableError",
    "resolve_ripgrep_config",
    "get_ripgrep_install_hint",
    "get_ripgrep_status",
    "rip_grep",
    "rip_grep_stream",
    "count_files_rounded_rg",
    # shell_quote
    "shell_quote",
    "shell_join",
    "shell_split",
    "escape_shell_special",
    # sleep
    "sleep",
    "with_timeout",
    # token_budget
    "parse_token_budget",
    "find_token_budget_positions",
    "get_budget_continuation_message",
    # uuid
    "validate_uuid",
    "generate_uuid",
    "create_agent_id",
    # validation
    "validate_array_of",
    "assert_non_empty_string",
    "assert_object",
    "assert_function",
    "assert_positive_int",
    "assert_in_range",
    # version
    "NIAHARNESS_RELEASES_URL",
    "PUBLIC_BUILD_VERSION",
    "normalize_public_version",
    "get_release_tag_url",
    "get_public_build_version",
]
