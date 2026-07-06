/**
 * AG-UI integration for the OpenAI Agents JS SDK.
 *
 * The adapter translates `Runner.run(agent, { stream: true })` stream events
 * into AG-UI protocol events. Call `adapter.run(input)` and subscribe to the
 * resulting AG-UI event stream.
 *
 * @example
 * ```typescript
 * import { OpenAIAgentsAdapter } from "@ag-ui/openai-agents";
 *
 * const adapter = new OpenAIAgentsAdapter({ model: "gpt-5" });
 * const events$ = adapter.run(input);
 * ```
 */

import { AbstractAgent } from "@ag-ui/client";
import { Observable, EMPTY } from "rxjs";
import type { BaseEvent, RunAgentInput } from "@ag-ui/core";

import { DEFAULT_MODEL, ENV } from "./config";
import type { OpenAIAgentsAdapterConfig, ProcessedEvent } from "./types";

/**
 * Error thrown when the adapter cannot construct a valid OpenAI client.
 */
export class OpenAIAgentsConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OpenAIAgentsConfigError";
  }
}

export class OpenAIAgentsAdapter extends AbstractAgent {
  private static readonly DEFAULT_MAX_STATES = 1000;
  private static readonly DEFAULT_STATE_TTL_MS = 30 * 60 * 1000;

  protected config: OpenAIAgentsAdapterConfig;

  constructor(config: OpenAIAgentsAdapterConfig = {}) {
    super(config);
    this.config = config;
  }

  public clone(): OpenAIAgentsAdapter {
    const cloned = super.clone() as OpenAIAgentsAdapter;
    cloned.config = { ...this.config };
    return cloned;
  }

  /**
   * Resolve the effective model name: explicit config > env > default.
   */
  protected resolveModel(): string {
    return (
      this.config.model ??
      (typeof process !== "undefined" ? process.env?.[ENV.MODEL] : undefined) ??
      DEFAULT_MODEL
    );
  }

  /**
   * Resolve the effective API key: explicit config > env.
   * Throws `OpenAIAgentsConfigError` when no key is available.
   */
  protected resolveApiKey(): string {
    const key =
      this.config.apiKey ??
      (typeof process !== "undefined" ? process.env?.[ENV.API_KEY] : undefined);
    if (!key) {
      throw new OpenAIAgentsConfigError(
        `OpenAI API key is required. Set the ${ENV.API_KEY} environment variable or pass \`apiKey\` to the adapter config.`,
      );
    }
    return key;
  }

  /**
   * Resolve the effective base URL: explicit config > env.
   */
  protected resolveBaseUrl(): string | undefined {
    return (
      this.config.baseURL ??
      (typeof process !== "undefined" ? process.env?.[ENV.BASE_URL] : undefined)
    );
  }

  /**
   * Validate the config: `baseURL` and `openAIClient` are mutually exclusive
   * (SDK constraint). Public so callers can fail-fast before a run.
   */
  public validateConfig(): void {
    if (this.config.openAIClient && (this.config.baseURL || this.config.apiKey)) {
      throw new OpenAIAgentsConfigError(
        "`openAIClient` cannot be combined with `apiKey` or `baseURL`.",
      );
    }
  }

  // Phase 1 stub: full streaming implementation lands in Phase 2.
  run(_input: RunAgentInput): Observable<BaseEvent> {
    return EMPTY as unknown as Observable<ProcessedEvent>;
  }
}
