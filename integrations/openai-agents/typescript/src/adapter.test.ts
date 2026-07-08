import { describe, it, expect, beforeEach, afterEach, vi, beforeAll } from "vitest";
import { EventType } from "@ag-ui/client";
import type { BaseEvent, RunAgentInput } from "@ag-ui/core";

import { OpenAIAgentsAdapter } from "./adapter";

/**
 * Collect all events emitted by `adapter.run(input)` into an array, resolving
 * on stream completion (RUN_FINISHED/RUN_ERROR or subscriber terminal).
 */
async function collectEvents(adapter: OpenAIAgentsAdapter, input: RunAgentInput): Promise<BaseEvent[]> {
  const events: BaseEvent[] = [];
  await new Promise<void>((resolve) => {
    const sub = adapter.run(input).subscribe({
      next: (e) => events.push(e as BaseEvent),
      error: () => resolve(),
      complete: () => resolve(),
    });
    // Safety timeout: resolve after a short delay if the stream never completes.
    setTimeout(() => { if (!sub.closed) sub.unsubscribe(); resolve(); }, 500);
  });
  return events;
}

/**
 * Mock the @openai/agents module at the import boundary so tests never need a
 * real OPENAI_API_KEY or network. The mock records how it was configured and
 * yields a configurable sequence of stream events.
 */

type StreamEvent =
  | { type: "raw_model_stream_event"; data: any }
  | { type: "run_item_stream_event"; name: string; item: any }
  | { type: "agent_updated_stream_event"; agent: any };

let lastRun: {
  agentConfig: Record<string, unknown>;
  input: unknown;
  providerCalls: Record<string, number>;
  setClientCalledWith?: unknown;
  setKeyCalledWith?: string;
  apiMode?: string;
  tracingDisabled?: boolean;
};

function baseInput(overrides: Partial<RunAgentInput> = {}): RunAgentInput {
  return {
    threadId: "thread-1",
    runId: "run-1",
    messages: [],
    tools: [],
    context: [],
    state: {},
    forwardedProps: {},
    ...overrides,
  } as unknown as RunAgentInput;
}

// We install the mock per-test by overriding the dynamic import. Because the
// adapter imports `@openai/agents` dynamically inside runAsync, we use
// vi.mock with a factory that reads a mutable events ref.
const eventsRef: { events: StreamEvent[]; throwErr?: Error } = {
  events: [],
};
vi.mock("@openai/agents", () => {
  return {
    Agent: class {
      constructor(cfg: Record<string, unknown>) {
        lastRun.agentConfig = cfg;
      }
    },
    run: async (_agent: unknown, input: unknown, _opts: { stream: true }) => {
      lastRun.input = input;
      if (eventsRef.throwErr) throw eventsRef.throwErr;
      return {
        async *[Symbol.asyncIterator]() {
          for (const ev of eventsRef.events) yield ev;
        },
      };
    },
    setDefaultOpenAIClient: (c: unknown) => {
      lastRun.setClientCalledWith = c;
    },
    setDefaultOpenAIKey: (k: string) => {
      lastRun.setKeyCalledWith = k;
    },
    setOpenAIAPI: (v: string) => {
      lastRun.apiMode = v;
    },
    setTracingDisabled: (d: boolean) => {
      lastRun.tracingDisabled = d;
    },
  };
});

// openai is imported dynamically by configureProvider for the LiteLLM path.
vi.mock("openai", () => {
  return {
    default: class {
      opts: unknown;
      constructor(opts: unknown) {
        this.opts = opts;
      }
    },
  };
});

beforeAll(() => {
  // initialize lastRun so the mock's closure has something to write to
  lastRun = {
    agentConfig: {},
    input: undefined,
    providerCalls: {},
  } as any;
});

describe("OpenAIAgentsAdapter — run lifecycle", () => {
  const origEnv = { ...process.env };
  beforeEach(() => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_BASE_URL;
    delete process.env.OPENAI_MODEL;
    eventsRef.events = [];
    eventsRef.throwErr = undefined;
    lastRun = { agentConfig: {}, input: undefined, providerCalls: {} } as any;
  });
  afterEach(() => {
    process.env = { ...origEnv };
  });

  it("emits RUN_STARTED → message events → MESSAGES_SNAPSHOT → RUN_FINISHED", async () => {
    eventsRef.events = [
      { type: "raw_model_stream_event", data: { type: "output_text_delta", delta: "Hello" } },
      { type: "run_item_stream_event", name: "message_output_created", item: { rawItem: {} } },
    ];
    process.env.OPENAI_API_KEY = "sk-test";

    const adapter = new OpenAIAgentsAdapter({ model: "gpt-5" });
    const events = await collectEvents(adapter, baseInput());

    const types = events.map((e) => e.type);
    expect(types[0]).toBe(EventType.RUN_STARTED);
    expect(types[types.length - 1]).toBe(EventType.RUN_FINISHED);
    expect(types).toContain(EventType.TEXT_MESSAGE_START);
    expect(types).toContain(EventType.TEXT_MESSAGE_CONTENT);
    expect(types).toContain(EventType.TEXT_MESSAGE_END);
    expect(types).toContain(EventType.MESSAGES_SNAPSHOT);

    const runFinished = events.find((e) => e.type === EventType.RUN_FINISHED) as any;
    expect(runFinished.outcome).toEqual({ type: "success" });
  });

  it("emits a full tool-call lifecycle", async () => {
    eventsRef.events = [
      { type: "run_item_stream_event", name: "tool_called", item: { rawItem: { callId: "c1", name: "get_weather", arguments: '{"city":"SF"}' } } },
      { type: "run_item_stream_event", name: "tool_output", item: { rawItem: { callId: "c1", name: "get_weather" }, output: "sunny" } },
    ];
    process.env.OPENAI_API_KEY = "sk-test";

    const adapter = new OpenAIAgentsAdapter({ model: "gpt-5" });
    const events = await collectEvents(adapter, baseInput());

    const types = events.map((e) => e.type);
    expect(types).toEqual(
      expect.arrayContaining([
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_ARGS,
        EventType.TOOL_CALL_END,
        EventType.TOOL_CALL_RESULT,
      ]),
    );
    const result = events.find((e) => e.type === EventType.TOOL_CALL_RESULT) as any;
    expect(result.content).toBe("sunny");
    expect(result.toolCallId).toBe("c1");
  });

  it("emits RUN_ERROR when the SDK run throws", async () => {
    eventsRef.throwErr = new Error("boom");
    process.env.OPENAI_API_KEY = "sk-test";

    const adapter = new OpenAIAgentsAdapter({ model: "gpt-5" });
    const events = await collectEvents(adapter, baseInput());

    const errorEvent = events.find((e) => e.type === EventType.RUN_ERROR) as any;
    expect(errorEvent).toBeDefined();
    expect(errorEvent.message).toBe("boom");
  });

  it("converts input messages to SDK input", async () => {
    eventsRef.events = [];
    process.env.OPENAI_API_KEY = "sk-test";

    const adapter = new OpenAIAgentsAdapter({ model: "gpt-5" });
    await collectEvents(adapter, baseInput({
      messages: [
        { id: "m1", role: "user", content: "hi" } as any,
        { id: "m2", role: "assistant", content: "hello" } as any,
      ],
    }));

    expect(lastRun.input).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });

  it("builds the agent with configured model + instructions", async () => {
    eventsRef.events = [];
    process.env.OPENAI_API_KEY = "sk-test";

    const adapter = new OpenAIAgentsAdapter({
      model: "gpt-5-mini",
      instructions: "You are a pirate",
      agentId: "pirate-agent",
    });
    await collectEvents(adapter, baseInput());

    expect(lastRun.agentConfig).toMatchObject({
      name: "pirate-agent",
      model: "gpt-5-mini",
      instructions: "You are a pirate",
    });
  });
});

describe("OpenAIAgentsAdapter — LiteLLM provider wiring", () => {
  const origEnv = { ...process.env };
  beforeEach(() => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_BASE_URL;
    eventsRef.events = [];
    lastRun = { agentConfig: {}, input: undefined, providerCalls: {} } as any;
  });
  afterEach(() => {
    process.env = { ...origEnv };
  });

  it("uses Chat Completions mode + disables tracing when baseURL is set", async () => {
    eventsRef.events = [];
    const adapter = new OpenAIAgentsAdapter({
      apiKey: "sk-test",
      baseURL: "https://litellm.example.com/v1",
    });
    await collectEvents(adapter, baseInput());

    expect(lastRun.apiMode).toBe("chat_completions");
    expect(lastRun.tracingDisabled).toBe(true);
    expect(lastRun.setClientCalledWith).toBeDefined();
  });

  it("sets the default key on the default cloud path", async () => {
    eventsRef.events = [];
    const adapter = new OpenAIAgentsAdapter({ apiKey: "sk-test" });
    await collectEvents(adapter, baseInput());

    expect(lastRun.setKeyCalledWith).toBe("sk-test");
    expect(lastRun.apiMode).toBeUndefined();
    expect(lastRun.tracingDisabled).toBeUndefined();
  });

  it("accepts a preconfigured openAIClient without baseURL", async () => {
    eventsRef.events = [];
    const fakeClient = { custom: true };
    const adapter = new OpenAIAgentsAdapter({ openAIClient: fakeClient });
    await collectEvents(adapter, baseInput());

    expect(lastRun.setClientCalledWith).toBe(fakeClient);
    expect(lastRun.tracingDisabled).toBe(true);
    // must NOT set chat_completions mode automatically (caller owns the client)
    expect(lastRun.apiMode).toBeUndefined();
  });
});
