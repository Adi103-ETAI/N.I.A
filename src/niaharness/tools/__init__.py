"""Built-in tool registration."""

from niaharness.tools.ask_user_question_tool import AskUserQuestionTool
from niaharness.tools.agent_tool import AgentTool
from niaharness.tools.BashTool import BashTool
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from niaharness.tools.brief_tool import BriefTool
from niaharness.tools.config_tool import ConfigTool
from niaharness.tools.cron_create_tool import CronCreateTool
from niaharness.tools.cron_delete_tool import CronDeleteTool
from niaharness.tools.cron_list_tool import CronListTool
from niaharness.tools.cron_toggle_tool import CronToggleTool
from niaharness.tools.enter_plan_mode_tool import EnterPlanModeTool
from niaharness.tools.enter_worktree_tool import EnterWorktreeTool
from niaharness.tools.exit_plan_mode_tool import ExitPlanModeTool
from niaharness.tools.exit_worktree_tool import ExitWorktreeTool
from niaharness.tools.FileEditTool import FileEditTool
from niaharness.tools.FileReadTool import FileReadTool
from niaharness.tools.FileWriteTool import FileWriteTool
from niaharness.tools.glob_tool import GlobTool
from niaharness.tools.grep_tool import GrepTool
from niaharness.tools.list_mcp_resources_tool import ListMcpResourcesTool
from niaharness.tools.lsp_tool import LspTool
from niaharness.tools.mcp_auth_tool import McpAuthTool
from niaharness.tools.mcp_tool import McpToolAdapter
from niaharness.tools.notebook_edit_tool import NotebookEditTool
from niaharness.tools.read_mcp_resource_tool import ReadMcpResourceTool
from niaharness.tools.remote_trigger_tool import RemoteTriggerTool
from niaharness.tools.send_message_tool import SendMessageTool
from niaharness.tools.skill_tool import SkillTool
from niaharness.tools.sleep_tool import SleepTool
from niaharness.tools.task_create_tool import TaskCreateTool
from niaharness.tools.task_get_tool import TaskGetTool
from niaharness.tools.task_list_tool import TaskListTool
from niaharness.tools.task_output_tool import TaskOutputTool
from niaharness.tools.task_stop_tool import TaskStopTool
from niaharness.tools.task_update_tool import TaskUpdateTool
from niaharness.tools.team_create_tool import TeamCreateTool
from niaharness.tools.team_delete_tool import TeamDeleteTool
from niaharness.tools.todo_write_tool import TodoWriteTool
from niaharness.tools.tool_search_tool import ToolSearchTool
from niaharness.tools.web_fetch_tool import WebFetchTool
from niaharness.tools.web_search_tool import WebSearchTool


def create_default_tool_registry(mcp_manager=None) -> ToolRegistry:
    """Return the default built-in tool registry."""
    registry = ToolRegistry()
    for tool in (
        BashTool(),
        AskUserQuestionTool(),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        NotebookEditTool(),
        LspTool(),
        McpAuthTool(),
        GlobTool(),
        GrepTool(),
        SkillTool(),
        ToolSearchTool(),
        WebFetchTool(),
        WebSearchTool(),
        ConfigTool(),
        BriefTool(),
        SleepTool(),
        EnterWorktreeTool(),
        ExitWorktreeTool(),
        TodoWriteTool(),
        EnterPlanModeTool(),
        ExitPlanModeTool(),
        CronCreateTool(),
        CronListTool(),
        CronDeleteTool(),
        CronToggleTool(),
        RemoteTriggerTool(),
        TaskCreateTool(),
        TaskGetTool(),
        TaskListTool(),
        TaskStopTool(),
        TaskOutputTool(),
        TaskUpdateTool(),
        AgentTool(),
        SendMessageTool(),
        TeamCreateTool(),
        TeamDeleteTool(),
    ):
        registry.register(tool)
    if mcp_manager is not None:
        registry.register(ListMcpResourcesTool(mcp_manager))
        registry.register(ReadMcpResourceTool(mcp_manager))
        for tool_info in mcp_manager.list_tools():
            registry.register(McpToolAdapter(mcp_manager, tool_info))
    return registry


__all__ = [
    "BaseTool",
    "ToolExecutionContext",
    "ToolRegistry",
    "ToolResult",
    "create_default_tool_registry",
]
