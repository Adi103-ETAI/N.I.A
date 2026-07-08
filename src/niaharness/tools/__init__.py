"""Built-in tool registration."""

from niaharness.tools.ask_user_question_tool import AskUserQuestionTool
from niaharness.tools.agent_tool import AgentTool
from niaharness.tools.BashTool import BashTool
from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from niaharness.tools.brief_tool import BriefTool
from niaharness.tools.browser_tool import BrowserTool
from niaharness.tools.config_tool import ConfigTool
from niaharness.tools.computer_use import ComputerUseTool
from niaharness.tools.cron_create_tool import CronCreateTool
from niaharness.tools.cron_delete_tool import CronDeleteTool
from niaharness.tools.cron_list_tool import CronListTool
from niaharness.tools.cron_toggle_tool import CronToggleTool
from niaharness.tools.delegate_task_tool import DelegateTaskTool
from niaharness.tools.enter_plan_mode_tool import EnterPlanModeTool
from niaharness.tools.enter_worktree_tool import EnterWorktreeTool
from niaharness.tools.exit_plan_mode_tool import ExitPlanModeTool
from niaharness.tools.exit_worktree_tool import ExitWorktreeTool
from niaharness.tools.FileEditTool import FileEditTool
from niaharness.tools.FileReadTool import FileReadTool
from niaharness.tools.FileWriteTool import FileWriteTool
from niaharness.tools.glob_tool import GlobTool
from niaharness.tools.grep_tool import GrepTool
from niaharness.tools.image_generate_tool import ImageGenerateTool
from niaharness.tools.list_mcp_resources_tool import ListMcpResourcesTool
from niaharness.tools.lsp_tool import LspTool
from niaharness.tools.mcp_auth_tool import McpAuthTool
from niaharness.tools.mcp_tool import McpToolAdapter
from niaharness.tools.memory_tool import MemoryBatchedTool
from niaharness.tools.notebook_edit_tool import NotebookEditTool
from niaharness.tools.process_tool import ProcessTool
from niaharness.tools.cronjob_tool import CronjobTool
from niaharness.tools.read_mcp_resource_tool import ReadMcpResourceTool
from niaharness.tools.remote_trigger_tool import RemoteTriggerTool
from niaharness.tools.run_code_tool import RunCodeTool
from niaharness.tools.send_message_tool import SendMessageTool
from niaharness.tools.session_search_tool import SessionSearchTool
from niaharness.tools.skill_tool import SkillTool
from niaharness.tools.skill_hub_tool import SkillHubTool
from niaharness.tools.skill_manage_tool import SkillManageTool
from niaharness.tools.sleep_tool import SleepTool
from niaharness.tools.speak_tool import SpeakTool
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
from niaharness.tools.vision_analyze_tool import VisionAnalyzeTool
from niaharness.tools.web_fetch_tool import WebFetchTool
from niaharness.tools.web_search_tool import WebSearchTool
from niaharness.tools.nia_memory_tool import NiaMemoryTool
from niaharness.tools.nia_context_tool import NiaContextTool
from niaharness.tools.nia_voice_tool import NiaVoiceTool
from niaharness.tools.nia_session_tool import NiaSessionTool


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
        ImageGenerateTool(),
        SkillTool(),
        SkillManageTool(),
        SkillHubTool(),
        SessionSearchTool(),
        ToolSearchTool(),
        VisionAnalyzeTool(),
        WebFetchTool(),
        WebSearchTool(),
        ConfigTool(),
        ComputerUseTool(),
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
        CronjobTool(),
        DelegateTaskTool(),
        RemoteTriggerTool(),
        TaskCreateTool(),
        TaskGetTool(),
        TaskListTool(),
        TaskStopTool(),
        TaskOutputTool(),
        TaskUpdateTool(),
        ProcessTool(),
        AgentTool(),
        SendMessageTool(),
        TeamCreateTool(),
        TeamDeleteTool(),
        NiaMemoryTool(),
        MemoryBatchedTool(),
        NiaContextTool(),
        NiaVoiceTool(),
        NiaSessionTool(),
        # Hermes/Jarvis capability layer — interactive browser, code
        # execution, and text-to-speech.  See audit P1/P2/P3.
        BrowserTool(),
        RunCodeTool(),
        SpeakTool(),
    ):
        registry.register(tool)
    if mcp_manager is not None:
        registry.register(ListMcpResourcesTool(mcp_manager))
        registry.register(ReadMcpResourceTool(mcp_manager))
        for tool_info in mcp_manager.list_tools():
            registry.register(McpToolAdapter(mcp_manager, tool_info))
    return registry


def register_nia_tools(registry: ToolRegistry, memory: object, context: object, engine: object = None) -> None:
    """Wire NIA's memory, context, and engine instances into the registered NIA tools.

    Call this after creating the registry and initializing NIA's subsystems.
    """
    mem_tool = registry.get("nia_memory")
    if mem_tool is not None:
        mem_tool.set_memory(memory)
    ctx_tool = registry.get("nia_context")
    if ctx_tool is not None:
        ctx_tool.set_context(context)
    session_tool = registry.get("nia_session")
    if session_tool is not None and engine is not None:
        session_tool.set_engine(engine)


__all__ = [
    "BaseTool",
    "ToolExecutionContext",
    "ToolRegistry",
    "ToolResult",
    "create_default_tool_registry",
]
