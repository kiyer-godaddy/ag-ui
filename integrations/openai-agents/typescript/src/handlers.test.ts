import { describe, it, expect } from "vitest";
import { EventType } from "@ag-ui/client";

import {
  StreamContext,
  handleRawModelStreamEvent,
  handleRunItemStreamEvent,
  handleAgentUpdatedStreamEvent,
  closeOpenBlocks,
  buildMessagesSnapshot,
} from "./handlers";

/**
 * Helper: collect emitted events from a series of handler calls.
 */
function run(handler: (ctx: StreamContext) => any[], ctx = new StreamContext("run-1")) {
  return { events: handler(ctx), ctx };
}

describe("handlers — raw model stream events", () => {
  it("opens a text message and emits CONTENT for output_text_delta", () => {
    const ctx = new StreamContext("run-1");
    const e1 = handleRawModelStreamEvent({ type: "output_text_delta", delta: "Hello" }, ctx);
    expect(e1).toEqual([
      { type: EventType.TEXT_MESSAGE_START, messageId: "run-1", role: "assistant" },
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "run-1", delta: "Hello" },
    ]);
    expect(ctx.messageOpen).toBe(true);
    expect(ctx.currentMessageId).toBe("run-1");

    const e2 = handleRawModelStreamEvent({ type: "output_text_delta", delta: " world" }, ctx);
    expect(e2).toEqual([
      { type: EventType.TEXT_MESSAGE_CONTENT, messageId: "run-1", delta: " world" },
    ]);
  });

  it("emits TOOL_CALL_ARGS deltas only after the call is open", () => {
    const ctx = new StreamContext("run-1");
    // arg delta before tool_called → buffered, not emitted
    const e1 = handleRawModelStreamEvent(
      { type: "model", event: { type: "response.function_call_arguments.delta", item_id: "call_a", delta: '{"city":' } },
      ctx,
    );
    expect(e1).toEqual([]);
    expect(ctx.toolCallArgs.get("call_a")).toBe('{"city":');

    // open the call
    const e2 = handleRunItemStreamEvent("tool_called", {
      rawItem: { callId: "call_a", name: "get_weather", arguments: "" },
    }, ctx);
    expect(e2[0]).toMatchObject({ type: EventType.TOOL_CALL_START, toolCallId: "call_a", toolCallName: "get_weather" });
    // buffered args flushed as one ARGS delta
    expect(e2[1]).toMatchObject({ type: EventType.TOOL_CALL_ARGS, toolCallId: "call_a", delta: '{"city":' });
    expect(e2[2]).toMatchObject({ type: EventType.TOOL_CALL_END, toolCallId: "call_a" });

    // subsequent arg deltas stream through
    const e3 = handleRawModelStreamEvent(
      { type: "model", event: { type: "response.function_call_arguments.delta", item_id: "call_a", delta: '"SF"}' } },
      ctx,
    );
    expect(e3).toEqual([
      { type: EventType.TOOL_CALL_ARGS, toolCallId: "call_a", delta: '"SF"}' },
    ]);
  });

  it("emits REASONING_MESSAGE_START + CONTENT for reasoning summary deltas", () => {
    const ctx = new StreamContext("run-1");
    const e1 = handleRawModelStreamEvent(
      { type: "model", event: { type: "response.reasoning_summary_text.delta", item_id: "rsn_1", delta: "Thinking" } },
      ctx,
    );
    expect(e1).toEqual([
      { type: EventType.REASONING_MESSAGE_START, messageId: "rsn_1", role: "reasoning" },
      { type: EventType.REASONING_MESSAGE_CONTENT, messageId: "rsn_1", delta: "Thinking" },
    ]);
    expect(ctx.reasoningOpen).toBe(true);
    expect(ctx.currentReasoningId).toBe("rsn_1");
  });

  it("ignores unknown raw event types", () => {
    const ctx = new StreamContext("run-1");
    expect(handleRawModelStreamEvent({ type: "model", event: { type: "response.refusal.delta" } }, ctx)).toEqual([]);
    expect(handleRawModelStreamEvent({ type: "response_started" }, ctx)).toEqual([]);
    expect(handleRawModelStreamEvent(null, ctx)).toEqual([]);
  });
});

describe("handlers — run item events", () => {
  it("tool_called emits START / ARGS(full) / END when no streamed deltas", () => {
    const ctx = new StreamContext("run-1");
    const e = handleRunItemStreamEvent("tool_called", {
      rawItem: { callId: "call_b", name: "search", arguments: '{"q":"x"}' },
    }, ctx);
    expect(e).toEqual([
      { type: EventType.TOOL_CALL_START, toolCallId: "call_b", toolCallName: "search" },
      { type: EventType.TOOL_CALL_ARGS, toolCallId: "call_b", delta: '{"q":"x"}' },
      { type: EventType.TOOL_CALL_END, toolCallId: "call_b" },
    ]);
    expect(ctx.openToolCalls.has("call_b")).toBe(true);
  });

  it("tool_called with no args emits START + END only", () => {
    const ctx = new StreamContext("run-1");
    const e = handleRunItemStreamEvent("tool_called", {
      rawItem: { callId: "call_c", name: "noop", arguments: "" },
    }, ctx);
    expect(e).toEqual([
      { type: EventType.TOOL_CALL_START, toolCallId: "call_c", toolCallName: "noop" },
      { type: EventType.TOOL_CALL_END, toolCallId: "call_c" },
    ]);
  });

  it("tool_output emits TOOL_CALL_RESULT with stringified output", () => {
    const ctx = new StreamContext("run-1");
    handleRunItemStreamEvent("tool_called", {
      rawItem: { callId: "call_d", name: "lookup", arguments: "{}" },
    }, ctx);
    const e = handleRunItemStreamEvent("tool_output", {
      rawItem: { callId: "call_d", name: "lookup" },
      output: { ok: true, count: 3 },
    }, ctx);
    expect(e).toHaveLength(1);
    expect(e[0]).toMatchObject({
      type: EventType.TOOL_CALL_RESULT,
      toolCallId: "call_d",
      messageId: "call_d",
      role: "tool",
    });
    expect((e[0] as any).content).toBe(JSON.stringify({ ok: true, count: 3 }));
  });

  it("tool_output with string output passes it through unchanged", () => {
    const ctx = new StreamContext("run-1");
    const e = handleRunItemStreamEvent("tool_output", {
      rawItem: { callId: "call_e" },
      output: "sunny",
    }, ctx);
    expect((e[0] as any).content).toBe("sunny");
  });

  it("message_output_created closes open text + reasoning blocks", () => {
    const ctx = new StreamContext("run-1");
    handleRawModelStreamEvent({ type: "output_text_delta", delta: "hi" }, ctx);
    handleRawModelStreamEvent(
      { type: "model", event: { type: "response.reasoning_summary_text.delta", item_id: "r1", delta: "hmm" } },
      ctx,
    );
    const e = handleRunItemStreamEvent("message_output_created", { rawItem: {} }, ctx);
    // closes reasoning first, then text message
    expect(e).toEqual([
      { type: EventType.REASONING_MESSAGE_END, messageId: "r1" },
      { type: EventType.TEXT_MESSAGE_END, messageId: "run-1" },
    ]);
    expect(ctx.messageOpen).toBe(false);
    expect(ctx.reasoningOpen).toBe(false);
  });

  it("reasoning_item_created closes an open reasoning block", () => {
    const ctx = new StreamContext("run-1");
    handleRawModelStreamEvent(
      { type: "model", event: { type: "response.reasoning_summary_text.delta", item_id: "r2", delta: "..." } },
      ctx,
    );
    const e = handleRunItemStreamEvent("reasoning_item_created", { rawItem: {} }, ctx);
    expect(e).toEqual([{ type: EventType.REASONING_MESSAGE_END, messageId: "r2" }]);
    expect(ctx.reasoningOpen).toBe(false);
  });

  it("approval/handoff/tool_search items map to CUSTOM events", () => {
    const ctx = new StreamContext("run-1");
    const approval = handleRunItemStreamEvent("tool_approval_requested", {
      rawItem: { callId: "call_f", name: "dangerous" },
      toJSON: () => ({ rawItem: { callId: "call_f" } }),
    }, ctx);
    expect(approval).toHaveLength(1);
    expect(approval[0]).toMatchObject({ type: EventType.CUSTOM, name: "tool_approval_requested" });

    const handoff = handleRunItemStreamEvent("handoff_occurred", {
      rawItem: {},
      toJSON: () => ({ to: "agent_b" }),
    }, ctx);
    expect(handoff[0]).toMatchObject({ type: EventType.CUSTOM, name: "handoff_occurred" });
  });
});

describe("handlers — agent updated", () => {
  it("emits a CUSTOM agent_updated event with the agent name", () => {
    const e = handleAgentUpdatedStreamEvent({ name: "Researcher" });
    expect(e).toEqual([
      { type: EventType.CUSTOM, name: "agent_updated", value: { name: "Researcher" } },
    ]);
  });
});

describe("handlers — end-of-stream cleanup", () => {
  it("closeOpenBlocks closes an open text message", () => {
    const ctx = new StreamContext("run-1");
    handleRawModelStreamEvent({ type: "output_text_delta", delta: "partial" }, ctx);
    const e = closeOpenBlocks(ctx);
    expect(e).toEqual([{ type: EventType.TEXT_MESSAGE_END, messageId: "run-1" }]);
    expect(ctx.messageOpen).toBe(false);
  });

  it("closeOpenBlocks is a no-op when nothing is open", () => {
    const ctx = new StreamContext("run-1");
    expect(closeOpenBlocks(ctx)).toEqual([]);
  });

  it("buildMessagesSnapshot concatenates input + accumulated", () => {
    const input = [{ id: "m1", role: "user", content: "hi" }] as any;
    const accumulated = [{ id: "m2", role: "assistant", content: "hello" }] as any;
    const out = buildMessagesSnapshot(input, accumulated);
    expect(out).toHaveLength(2);
    expect(out[0].id).toBe("m1");
    expect(out[1].id).toBe("m2");
  });
});
