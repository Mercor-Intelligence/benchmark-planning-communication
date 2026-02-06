"""
Additional tools for SWE-Bench-Ext.

This module provides benchmark-specific tools that extend the base
Lighthouse tool set for the SWE-Bench-Ext benchmark.

Tools:
- AskQuestionTool: Enables agent to ask clarifying questions to a simulated user
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from lighthouse.core.registry import tool
from lighthouse.core.tools.base_tool import BaseTool, BaseHyperparameters, ToolResult
from lighthouse.core.harness.base_sandbox import BaseSandbox

from lighthouse.core.benchmark_tasks.base_benchmark_task import BaseBenchmarkTask
from lighthouse.core.options import HarnessExecutionOptions
import litellm

from .kb_filters import filter_knowledge_base_for_responder


# =============================================================================
# Constants
# =============================================================================

# Default path for storing chat history in the sandbox
ASK_QUESTION_HISTORY_PATH = "/tmp/ask_question_history.json"

# Default responder model
DEFAULT_RESPONDER_MODEL = "anthropic/claude-sonnet-4-20250514"

# Expert responder prompt - provides detailed, helpful responses
EXPERT_RESPONDER_PROMPT = """You are an expert user who deeply understands the problem being solved. 
You are responding to questions from a developer who is implementing a solution.

Your role:
1. Answer questions accurately based on the problem context and your knowledge base
2. Provide helpful clarifications when asked about requirements or expected behavior
3. Guide the developer without giving away the complete solution
4. Be concise but thorough - give enough detail to be helpful
5. If asked about something not covered in the context, say you're not sure

Important guidelines:
- Stay in character as the user who reported the issue or requested the feature
- Don't reveal implementation details unless specifically asked
- Focus on what the solution should do, not how to implement it
- If the question is unclear, ask for clarification
"""

EXPERT_RESPONDER_PROMPT_ENHANCED = """You are an expert user (highly technical) who deeply understands the problem being solved.
You are responding to questions from a developer who is implementing a solution.

Rules:
1. Answer ONLY using the context provided below (problem, request, interface, requirements, knowledge base).
2. Be direct and actionable, but do NOT write code or propose specific patches.
3. If asked about implementation details, explain expected behavior and constraints instead.
4. If something is not covered in the context, say you are not sure.
5. Stay in character as the user/requester.
"""

# Novice responder prompt - provides less detailed responses
NOVICE_RESPONDER_PROMPT = """You are a user who has reported a problem or requested a feature.
You are responding to questions from a developer who is implementing a solution.

Your role:
1. Answer questions based on what you know about the problem
2. You may not understand all technical details
3. Focus on describing the behavior you expect from the user's perspective
4. Be honest when you're not sure about something

Important guidelines:
- Stay in character as a non-technical user
- Describe what you want in terms of outcomes, not implementation
- If asked about edge cases you didn't consider, say you're not sure
- Be helpful but don't pretend to know more than you do
"""

NOVICE_RESPONDER_PROMPT_ENHANCED = """You are a non-technical user who reported a problem or requested a feature.
You are responding to questions from a developer.

Rules:
1. Answer ONLY using the context provided below.
2. Do NOT mention file names/paths, function/class names, stack traces, or code snippets.
3. Describe expected behavior in plain language and user outcomes.
4. If asked about technical internals, say you are not sure and describe what you observe/expect.
5. Stay in character as a non-technical user.
"""


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


# =============================================================================
# Hyperparameters
# =============================================================================

@dataclass
class AskQuestionHyperparameters(BaseHyperparameters):
    """
    Hyperparameters for AskQuestionTool.
    
    Attributes:
        history_path: Path to store chat history in the sandbox.
        responder_model: LLM model to use for simulating user responses.
        responder_api_key: API key for the responder model (optional).
        responder_type: Type of responder ("expert" or "novice").
        problem_statement: The problem description for context.
        prompt_statement: The user request/prompt for context.
        knowledge_base: Additional knowledge base content for context.
        interface_spec: Interface specifications for context.
        requirements: List of requirements for context.
    """
    history_path: str = ASK_QUESTION_HISTORY_PATH
    responder_model: str = DEFAULT_RESPONDER_MODEL
    responder_api_key: str = ""
    responder_type: str = "expert"  # "expert" or "novice"
    use_enhanced_prompts: bool = True
    
    # Context from the task (populated by from_benchmark_task)
    problem_statement: str = ""
    prompt_statement: str = ""
    knowledge_base: str = ""
    interface_spec: str = ""
    requirements: List[str] = field(default_factory=list)


# =============================================================================
# Tool Implementation
# =============================================================================

@tool("ask_question")
class AskQuestionTool(BaseTool):
    """
    User interaction tool for asking clarifying questions.
    
    This tool enables agents to communicate with a simulated user when they need
    additional information or clarification about the problem they're solving.
    
    The simulated user is powered by an LLM that has access to the problem context
    and knowledge base, and responds based on a configurable persona (expert/novice).
    
    Best practices for agents using this tool:
    - Use BEFORE starting to solve the problem to clarify requirements
    - Ask when the problem statement is ambiguous
    - Request specific information about edge cases or expected behavior
    - Confirm understanding before implementing solutions
    
    Hyperparameters:
        history_path: Path for chat history persistence.
        responder_model: LLM model for simulated responses.
        responder_api_key: API key for the responder model.
        responder_type: "expert" or "novice" persona.
        problem_statement: Problem context for the responder.
        prompt_statement: User request context.
        knowledge_base: Additional knowledge for the responder.
        interface_spec: Interface specifications.
        requirements: List of requirements.
    """
    
    name = "ask_question"
    description = "Ask a clarifying question to the user about the problem or requirements"
    hyperparameters_class = AskQuestionHyperparameters
    
    # Optional system prompt addition for the agent
    prompt_addition = """
## Ask Question Tool

Use the ask_question tool to communicate with the user when you need clarification.

Best practices:
1. HIGHLY RECOMMENDED to use this before starting to solve the problem
2. Ask when requirements are ambiguous or unclear
3. Use when you get stuck and need additional context
4. Request specific information about edge cases or expected behavior
5. Be specific and clear in your questions
6. Ask one focused question at a time or group related questions together
"""
    
    # Type hint for hyperparameters
    hyperparameters: AskQuestionHyperparameters
    
    @classmethod
    def from_benchmark_task(
        cls,
        task: BaseBenchmarkTask,
        get_sandbox_func: Callable[[], BaseSandbox],
        harness_options: HarnessExecutionOptions,
    ) -> "AskQuestionTool":
        """
        Create AskQuestionTool configured for a benchmark task.
        
        This extracts the relevant context (problem statement, knowledge base, etc.)
        from the task instance to provide to the simulated responder.
        
        Args:
            task: The benchmark task to configure the tool for.
            get_sandbox_func: Callable that returns the sandbox environment.
            responder_model: LLM model for responses (default: claude-sonnet).
            responder_api_key_env_var: Environment variable for API key.
            responder_type: "expert" or "novice" persona.
            
        Returns:
            Configured AskQuestionTool instance.
        """


        responder_model = harness_options.tool_options.get("ask_question", {}).get(
            "responder_model", harness_options.model
        )

        responder_api_key = harness_options.tool_options.get("ask_question", {}).get(
            "responder_api_key", ""
        )

        responder_type = harness_options.tool_options.get("ask_question", {}).get("responder_type")
        if not responder_type:
            responder_type = os.environ.get("RESPONDER_TYPE", "expert")

        # Allow enabling enhanced prompts via tool options, falling back to env var
        use_enhanced_prompts = harness_options.tool_options.get("ask_question", {}).get(
            "use_enhanced_prompts"
        )
        if use_enhanced_prompts is None:
            use_enhanced_prompts = _env_bool("USE_ENHANCED_PROMPTS", default=False)
        # Extract context from the task instance
        task_instance = task.task_instance
        
        # Get fields that may or may not exist on the task instance
        problem_statement = getattr(task_instance, "problem_statement", "")
        prompt_statement = getattr(task_instance, "prompt_statement", problem_statement)
        knowledge_base = getattr(task_instance, "knowledge_base", "")
        interface_spec = getattr(task_instance, "interface", "")
        requirements = getattr(task_instance, "requirements", [])
        
        # Ensure requirements is a list
        if isinstance(requirements, str):
            requirements = [requirements] if requirements else []
        
        hyperparams = AskQuestionHyperparameters(
            responder_model=responder_model or DEFAULT_RESPONDER_MODEL,
            responder_api_key=responder_api_key or "",
            responder_type=responder_type,
            use_enhanced_prompts=bool(use_enhanced_prompts),
            problem_statement=problem_statement,
            prompt_statement=prompt_statement,
            knowledge_base=knowledge_base,
            interface_spec=interface_spec,
            requirements=list(requirements),
        )
        
        return cls(get_sandbox_func=get_sandbox_func, hyperparameters=hyperparams)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the simulated responder."""
        hp = self.hyperparameters
        
        use_enhanced = bool(getattr(hp, "use_enhanced_prompts", False))

        responder_type = (hp.responder_type or "expert").strip().lower()
        if responder_type == "novice":
            base_prompt = (
                NOVICE_RESPONDER_PROMPT_ENHANCED
                if use_enhanced
                else NOVICE_RESPONDER_PROMPT
            )
        else:
            base_prompt = (
                EXPERT_RESPONDER_PROMPT_ENHANCED
                if use_enhanced
                else EXPERT_RESPONDER_PROMPT
            )

        filtered_kb = filter_knowledge_base_for_responder(
            kb=hp.knowledge_base,
            responder_type=responder_type,
        )
        
        # Format requirements as numbered list
        requirements_text = ""
        if hp.requirements:
            requirements_text = "\n".join(
                f"{i + 1}. {req}" for i, req in enumerate(hp.requirements)
            )
        
        # Build full system prompt with context
        system_prompt = f"""{base_prompt}

## PROBLEM CONTEXT

### Problem Description
{hp.problem_statement}

### User Request
{hp.prompt_statement}

### Interface Specifications
{hp.interface_spec}

### Requirements
{requirements_text}

## KNOWLEDGE BASE

{filtered_kb}"""
        
        return system_prompt
    
    async def _load_chat_history(self, sandbox: BaseSandbox) -> List[Dict[str, str]]:
        """Load chat history from the sandbox."""
        history_path = self.hyperparameters.history_path
        
        result = await sandbox.exec(f"cat '{history_path}' 2>/dev/null || echo '[]'")
        
        if result.success and result.stdout.strip():
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return []
        return []
    
    async def _save_chat_history(
        self, sandbox: BaseSandbox, history: List[Dict[str, str]]
    ) -> bool:
        """Save chat history to the sandbox using base64 encoding."""
        history_path = self.hyperparameters.history_path
        history_json = json.dumps(history)
        encoded = base64.b64encode(history_json.encode('utf-8')).decode('ascii')
        
        result = await sandbox.exec(
            f"echo '{encoded}' | base64 -d > '{history_path}'"
        )
        return result.success
    
    async def execute_tool(self, user_message: str) -> ToolResult:
        """
        Ask a question to the simulated user.
        
        This tool enables direct communication with a simulated user who has
        context about the problem being solved.
        
        Important usage notes:
        1. HIGHLY RECOMMENDED to use this before starting to solve the problem
        2. Ask clarifying questions when the problem statement is ambiguous
        3. Use when you get stuck and need additional context or guidance
        4. Request specific information about requirements or expected behavior
        5. Confirm your understanding before implementing solutions
        
        Best practices:
        - Be specific and clear in your questions
        - Ask one focused question at a time or group related questions together
        - Explain why you need the information to help get better answers
        
        Args:
            user_message: Your question or message to the user.
            
        Returns:
            ToolResult with the simulated user's response.
        """
        try:
            # Import litellm here to avoid import errors if not installed
            hp = self.hyperparameters
            sandbox = self.get_sandbox_func()
            
            # Load existing chat history
            chat_history = await self._load_chat_history(sandbox)
            
            # Build system prompt with problem context
            system_prompt = self._build_system_prompt()
            
            # Add user message to history
            chat_history.append({"role": "user", "content": user_message})
            
            # Prepare messages for LLM
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(chat_history)
            

            # Make async LLM call using litellm
            response = await litellm.acompletion(
                model=hp.responder_model,
                messages=messages,
                api_key=hp.responder_api_key or None,
            )
            
            assistant_response = response.choices[0].message.content or ""
            
            # Add assistant response to history
            chat_history.append({"role": "assistant", "content": assistant_response})
            
            # Save updated history back to sandbox
            save_success = await self._save_chat_history(sandbox, chat_history)
            if not save_success:
                # Non-fatal: response was still obtained
                pass
            
            return ToolResult(
                output=assistant_response,
                success=True,
                metadata={
                    "responder_model": hp.responder_model,
                    "responder_type": hp.responder_type,
                    "history_length": len(chat_history),
                },
            )
            
        except Exception as e:
            return ToolResult(
                output="",
                success=False,
                error=f"Error in ask_question: {str(e)}",
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": (
                        "Your question or message to the user. Be specific and clear "
                        "about what information you need."
                    ),
                },
            },
            "required": ["user_message"],
        }


__all__ = [
    "AskQuestionHyperparameters",
    "AskQuestionTool",
]
