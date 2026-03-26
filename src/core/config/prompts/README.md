# N.I.A. Prompt System

This directory contains all LLM system prompts for N.I.A. agents. Each prompt is a markdown file that defines behavior and guidelines for its respective agent.

## Available Prompts

### 1. `planner.md` - Mission Planner
**Agent**: MissionPlanner
**Role**: Strategic planning and scope classification
**Usage**: Analyzes user intent and creates execution plans

- Classifies tasks into required capability scopes
- Determines execution complexity (fast/standard/deep)
- Estimates agents and steps needed
- Provides precise scope selection (not over-declaring)

### 2. `supervisor.md` - Supervisor Agent
**Agent**: Supervisor
**Role**: General assistance and task coordination
**Usage**: Handles conversational requests and high-level coordination

- Answers general questions
- Provides system status and capabilities
- Coordinates between specialist agents
- Provides guidance and tutorials

### 3. `coder.md` - TARA Coder Agent
**Agent**: TARA
**Role**: Code generation and execution
**Usage**: Writes, tests, and executes code in sandboxes

- Generates code in multiple languages
- Executes in secure Docker containers
- Handles debugging and testing
- Ensures security and best practices

### 4. `researcher.md` - IRIS Researcher Agent
**Agent**: IRIS
**Role**: Data analysis and visual processing
**Usage**: Researches information and analyzes visual data

- Analyzes screenshots and visual data
- Conducts research and fact-finding
- Extracts text via OCR
- Synthesizes information
- Provides structured findings

### 5. `reviewer.md` - Code Reviewer Agent
**Agent**: Reviewer
**Role**: Code quality and security review
**Usage**: Reviews code for quality, security, and best practices

- Analyzes code structure and logic
- Checks for security vulnerabilities
- Evaluates performance
- Enforces coding standards
- Provides constructive feedback

## Usage

### Loading Prompts in Code

```python
from src.core.config.prompts import load_prompt

# Load a prompt
planner_prompt = load_prompt("planner")
supervisor_prompt = load_prompt("supervisor")
coder_prompt = load_prompt("coder")

# Use in your agent
response = llm.invoke([
    SystemMessage(content=load_prompt("supervisor")),
    HumanMessage(content=user_input),
])
```

### Listing Available Prompts

```python
from src.core.config.prompts import list_prompts, prompt_exists

# List all prompts
available = list_prompts()
# Output: ['coder', 'planner', 'researcher', 'reviewer', 'supervisor']

# Check if prompt exists
if prompt_exists("researcher"):
    prompt = load_prompt("researcher")
```

## Prompt Structure

Each prompt file contains:

1. **Title**: Agent name and role
2. **Capabilities**: What the agent can do
3. **Responsibilities**: What the agent is responsible for
4. **Guidelines**: How to approach tasks
5. **Process/Workflow**: Step-by-step procedures
6. **Examples**: Real interaction examples
7. **Standards**: Quality and technical standards
8. **Notes**: Special considerations

## Adding New Prompts

1. Create a new `.md` file in this directory:
```bash
touch src/core/config/prompts/new_agent.md
```

2. Add your prompt content following the standard structure

3. Load in code:
```python
from src.core.config.prompts import load_prompt
prompt = load_prompt("new_agent")
```

## Updating Prompts

Simply edit the markdown file:
```bash
vim src/core/config/prompts/planner.md
# Make changes
# Changes are loaded automatically on next plan() call!
```

## Design Philosophy

- ✅ **External Files**: Prompts not hardcoded in Python
- ✅ **Readable Format**: Markdown for easy reading and editing
- ✅ **Non-Developer Friendly**: Anyone can update prompts
- ✅ **Version Control**: Prompts tracked in git
- ✅ **Hot-Reload Ready**: Can be extended for dynamic loading
- ✅ **Modular**: Each agent has its own prompt
- ✅ **Examples-Driven**: Clear examples in each prompt
- ✅ **Standards-Based**: Quality guidelines included

## Prompt Characteristics

### Planner Prompt
- **Philosophy**: Precision over conservatism
- **Key Feature**: Scope examples table
- **Goal**: Accurate task classification

### Supervisor Prompt
- **Philosophy**: Professional yet friendly
- **Key Feature**: Decision tree for delegation
- **Goal**: User satisfaction and coordination

### Coder Prompt
- **Philosophy**: Security and quality first
- **Key Feature**: Sandbox execution guidelines
- **Goal**: Reliable code execution

### Researcher Prompt
- **Philosophy**: Accuracy and verification
- **Key Feature**: Analysis process flowchart
- **Goal**: Trustworthy information

### Reviewer Prompt
- **Philosophy**: Constructive and thorough
- **Key Feature**: Review checklist
- **Goal**: Code quality assurance

## Best Practices

1. **Keep it Simple**: Clear and concise language
2. **Use Examples**: Show expected behaviors
3. **Be Specific**: Provide concrete guidelines
4. **Version Control**: Track changes in git
5. **Test After Changes**: Verify behavior impact
6. **Document Updates**: Add comments about changes

## Prompt Loader Configuration

**Location**: `src/core/config/prompts.py`

Functions:
- `load_prompt(name, fallback=None)` - Load prompt from markdown
- `list_prompts()` - List available prompts
- `prompt_exists(name)` - Check if prompt exists

Features:
- Caching for performance
- Fallback text if file not found
- Automatic encoding (UTF-8)
- Error handling and logging

## Related Files

- `src/core/config/prompts.py` - Prompt loader module
- `src/agents/nia/planner.py` - Uses planner.md
- `src/agents/nia/supervisor.py` - Uses supervisor.md
- `src/agents/tara/` - Uses coder.md
- `src/agents/iris/` - Uses researcher.md

## Future Enhancements

- [ ] Hot-reload without restart
- [ ] Prompt versioning system
- [ ] A/B testing framework
- [ ] Prompt metrics and analytics
- [ ] User-custom prompts
- [ ] Prompt templates library
