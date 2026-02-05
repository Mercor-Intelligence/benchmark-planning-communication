"""
Reminder message constants and handlers for README and ask_question features.

Phase 1 Implementation (Constant Mode - Show Once at Start):
"""

from typing import Any, Optional

from lighthouse.core.benchmark_tasks.models import TaskStage

# =============================================================================
# PHASE 1: Hardcoded reminder message constants
# =============================================================================

README_REMINDER_LINE = "- Created/updated a README.md documenting the solution"
ASK_QUESTION_REMINDER_LINE = (
    "- Used ask_question() at least once to confirm requirements / unblock progress"
)


# =============================================================================
# PHASE 1: Custom handler with timing check (matches swe-bench-ext-harness)
# =============================================================================


class ConstantReminderWithTiming:
    """
    Handler that shows constant reminders with configurable frequency.
    Current default value: frequency=1 (every turn)
    """
    
    def __init__(self, reminder_items: list[str], frequency: int = 1):
        """
        Initialize with list of reminder strings to show.
        
        Args:
            reminder_items: List of reminder bullet points (e.g., ["- Created README...", ...])
            frequency: Show reminder every N turns (default: 1). Set to <= 0 to disable.
        """
        self.reminder_items = reminder_items
        self.frequency = frequency
        self.turn_count = 0  # Track number of on_continue calls
        
        self.footer_text = "\n\n✅ Reminders:\n" + "\n".join(reminder_items)
    
    async def on_stage_start(
        self,
        *,
        stage: TaskStage,
        trajectory: Any,
        metadata: dict[str, Any],
    ) -> Optional[str]:
        """
        Show reminders at stage start.
        
        Args:
            stage: Current task stage
            trajectory: List of messages (inspect-ai state.messages)
            metadata: Additional metadata
            
        Returns:
            Footer text to show at stage start
        """
        # For "every turn" reminders, avoid injecting at stage start so this
        # doesn't look like "after every stage". The per-turn behavior is driven
        # by on_continue().
        self.turn_count = 0  # Reset counter for new stage
        return None
    
    async def on_continue(
        self,
        *,
        stage: TaskStage,
        trajectory: Any,
        metadata: dict[str, Any],
    ) -> Optional[str]:
        """
        Show reminders periodically (every N turns if frequency > 0).
        
        This provides gentle periodic nudges without overwhelming the agent.
        
        Returns:
            - Footer text every Nth turn (if frequency > 0)
            - None otherwise (adapter.py converts to True for react())
        """
        # Disabled when frequency <= 0
        if self.frequency <= 0:
            return None
        
        self.turn_count += 1
        
        # Show reminder every `frequency` turns
        if self.turn_count % self.frequency == 0:
            return self.footer_text.strip()
        
        return None
    
    async def on_stage_end(
        self,
        *,
        stage: TaskStage,
        trajectory: Any,
        metadata: dict[str, Any],
    ) -> Optional[str]:
        """
        No reminders at stage end.
        
        Returns:
            None (no message injection)
        """
        return None

# =============================================================================
# PHASE 2: Staged mode handler classes (not implemented yet)
# =============================================================================

# When implementing staged mode, add handler classes here that follow the same
# pattern as ConstantReminderWithTiming but with additional state checking:
#
# class StagedReadmeReminder:
#     """
#     Check conversation for README mentions and inject contextual reminders.
#     Matches swe-bench-ext-harness create_readme_reminder_handler() behavior.
#     """
#     async def on_continue(
#         self,
#         *,
#         stage: TaskStage,
#         trajectory: Any,
#         metadata: dict[str, Any],
#     ) -> Optional[str]:
#         # 1. Check if agent has started responding (timing check)
#         # 2. Check if README mentioned in trajectory
#         # 3. Return appropriate reminder string based on message count and context
#         # 4. Return None if no reminder needed (continues agent without message)
#         pass
#
# class StagedAskQuestionReminder:
#     """
#     Track ask_question usage and inject escalating reminders.
#     Matches swe-bench-ext-harness create_ask_question_reminder_handler() behavior.
#     """
#     async def on_continue(
#         self,
#         *,
#         stage: TaskStage,
#         trajectory: Any,
#         metadata: dict[str, Any],
#     ) -> Optional[str]:
#         # 1. Check if agent has started responding (timing check)
#         # 2. Check if ask_question was called in trajectory
#         # 3. Return escalating reminders based on message count
#         # 4. Return None if no reminder needed
#         pass
