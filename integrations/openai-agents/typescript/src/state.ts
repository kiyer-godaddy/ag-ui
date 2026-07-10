/**
 * State helpers for the OpenAI Agents JS adapter.
 *
 * Phase 3 adds only the `hasState` predicate used to decide whether to inject
 * the `ag_ui_update_state` tool. Phase 4 adds state interception / snapshot
 * merging; Phase 5 adds the `RunState` pause/resume store.
 */

/**
 * Whether a state value is meaningful (non-null, non-undefined).
 *
 * The CopilotKit runtime sends `state: {}` even for agents that don't use a
 * shared state. We treat any object (including `{}`) as "has state" so the
 * state tool is injected and the model can populate it; `null`/`undefined`
 * mean no state is in play.
 *
 * (Mirrors the claude-agent-sdk `hasState` predicate so both integrations
 * agree on when state is active.)
 */
export function hasState(state: unknown): boolean {
  return state != null;
}
