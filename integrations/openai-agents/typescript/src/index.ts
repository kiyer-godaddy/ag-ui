/**
 * AG-UI integration for the OpenAI Agents JS SDK (TypeScript).
 *
 * The adapter manages the SDK run lifecycle internally — just call
 * `adapter.run(input)` and subscribe to the resulting AG-UI events.
 *
 * @example
 * ```typescript
 * import { OpenAIAgentsAdapter } from "@ag-ui/openai-agents";
 *
 * const adapter = new OpenAIAgentsAdapter({ model: "gpt-5" });
 * const events$ = adapter.run(input);
 * ```
 */

export { OpenAIAgentsAdapter, OpenAIAgentsConfigError } from "./adapter";
export type {
  OpenAIAgentsAdapterConfig,
  ProcessedEvent,
} from "./types";
export {
  STATE_MANAGEMENT_TOOL_NAME,
  DEFAULT_MODEL,
  ENV,
} from "./config";
