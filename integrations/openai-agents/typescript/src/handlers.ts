/**
 * Stream-event → AG-UI event translation for the OpenAI Agents JS adapter.
 *
 * Each handler is a pure function over an SDK stream event plus a `StreamContext`,
 * returning an array of AG-UI `BaseEvent`s to emit. Keeping this layer pure and
 * side-effect-free makes the mapping trivially unit-testable without the SDK.
 */

import { EventType } from "@ag-ui/client";
import type {
  BaseEvent,
  Message,
} from "@ag-ui/core";

import { outputToString } from "./utils";

/**
 * Per-run mutable state shared across handlers. Tracks which message / tool
 * call / reasoning block is currently open so deltas attach to the right IDs.
 */
export class StreamContext {
  /** AG-UI message id for the currently-streaming assistant text message, if any. */
  currentMessageId?: string;
  /** True once a TEXT_MESSAGE_START has been emitted for the current message. */
  messageOpen = false;

  /** AG-UI message id for the currently-streaming reasoning block, if any. */
  currentReasoningId?: string;
  /** True once a REASONING_MESSAGE_START has been emitted for the current block. */
  reasoningOpen = false;

  /** toolCallId → whether a TOOL_CALL_START has been emitted for that call. */
  openToolCalls = new Set<string>();

  /** toolCallId → accumulated argument-string deltas (so we can emit on done). */
  readonly toolCallArgs = new Map<string, string>();

  /** toolCallId → tool name, captured when the call item is first seen. */
  readonly toolCallNames = new Map<string, string>();

  constructor(
    /** Base id used to derive message ids when the SDK provides none. */
    public readonly runId: string,
  ) {}
}

/* ----------------------------- raw model events ---------------------------- */

/**
 * The shape of the raw model stream as exposed by `@openai/agents` v0.12.
 *
 * `RunRawModelStreamEvent.data` is a `ResponseStreamEvent` (see
 * `@openai/agents-core` `dist/types/helpers.d.ts`), which is the union
 * `StreamEvent` defined in `dist/types/protocol.d.ts`:
 *
 *   type StreamEvent =
 *     | { type: "output_text_delta"; delta: string }      // typed directly
 *     | { type: "response_done";     response: ... }       // typed directly
 *     | { type: "response_started" }                      // typed directly
 *     | { type: "model"; event: <raw Responses-API event> }; // passthrough
 *
 * Crucially, only `output_text_delta` is typed with a `delta` field. Every
 * other raw delta the Responses API emits — tool-call argument deltas,
 * reasoning-summary deltas, refusal deltas, etc. — arrives inside the
 * generic `{ type: "model", event }` passthrough as a raw
 * Responses-API event from the `openai` package (e.g.
 * `response.function_call_arguments.delta`,
 * `response.reasoning_summary_text.delta`).
 *
 * ⚠️ UPGRADE IMPACT: this is the part most likely to break on a
 * `@openai/agents` (or `openai`) bump:
 *   1. The `StreamEvent` union members may grow (e.g. a future SDK could type
 *      `tool_call_argument_delta` directly instead of routing it through the
 *      `model` passthrough). If so, the `data.type === "output_text_delta"`
 *      branch below would gain siblings and the `data.type === "model"`
 *      fallback would simply stop receiving those events.
 *   2. The raw `event.type` string literals come from the `openai` package's
 *      `ResponseStreamEvent` union (`openai/resources/responses/responses.d.ts`).
 *      They are a public API but have changed between major `openai` releases.
 *      NOTE: `@openai/agents` v0.12 bundles `openai@6.x` internally, while this
 *      package's own `import "openai"` may resolve to a different major (it
 *      declares `openai` as an optional peer/dev dep). The literals that
 *      actually flow through the passthrough at runtime are those of the
 *      `openai` version `@openai/agents` bundles — not necessarily the one this
 *      package resolves to — so treat these strings as a runtime contract, not
 *      something type-checkable from this package's `openai` types.
 *   3. The field names on the raw events (`item_id`, `delta`, `output_index`)
 *      are owned by the `openai` package, not `@openai/agents`.
 *
 * To catch drift early, `src/sdk-shapes.test.ts` pins these shapes two ways:
 *   - COMPILE-TIME guards fail the build if the SDK renames a class `type`
 *     discriminant, a run-item event name, the directly-typed
 *     `output_text_delta` shape, or the `model` passthrough wrapper.
 *   - RUNTIME contract tests pin the exact raw passthrough literals + fields
 *     (`item_id`, `delta`) as fixtures and assert this function maps each to
 *     the expected AG-UI event. If an `openai` major bump renames a literal or
 *     field, the fixture stops matching and the test fails — forcing a
 *     conscious update of both the test and this switch together.
 */
export function handleRawModelStreamEvent(
  data: any,
  ctx: StreamContext,
): BaseEvent[] {
  const events: BaseEvent[] = [];
  if (!data || typeof data !== "object") return events;

  // Direct text deltas — the only non-passthrough stream shape the SDK types
  // (see the StreamEvent union in @openai/agents-core protocol.d.ts).
  if (data.type === "output_text_delta" && typeof data.delta === "string") {
    if (!ctx.messageOpen) {
      ctx.currentMessageId = ctx.currentMessageId ?? ctx.runId;
      ctx.messageOpen = true;
      events.push({
        type: EventType.TEXT_MESSAGE_START,
        messageId: ctx.currentMessageId,
        role: "assistant",
      });
    }
    events.push({
      type: EventType.TEXT_MESSAGE_CONTENT,
      messageId: ctx.currentMessageId!,
      delta: data.delta,
    });
    return events;
  }

  // Passthrough "model" events wrap raw Responses-API events in `event`.
  // The wrapper shape `{ type: "model", event: <ResponseStreamEvent> }` is
  // `StreamEventGenericItem` in @openai/agents-core; the inner `event` is a
  // raw event from the `openai` package (see the doc comment above for why
  // this is upgrade-sensitive).
  const raw = data.type === "model" ? data.event : data;
  if (!raw || typeof raw !== "object") return events;

  // The `raw.type` literals below are the string discriminants of the
  // `openai` package's `ResponseStreamEvent` union. If a future `openai`
  // release renames them, the matching cases here go silent (deltas stop
  // mapping) — `sdk-shapes.test.ts` pins these literals to catch that.
  switch (raw.type) {
    case "response.function_call_arguments.delta": {
      // Shape: { type, item_id, delta, output_index, sequence_number }
      const callId = raw.item_id;
      if (typeof callId !== "string" || typeof raw.delta !== "string") break;
      // TOOL_CALL_START is emitted from the run_item `tool_called` event, which
      // carries the tool name. If args stream before the item (ordering varies),
      // we buffer them; TOOL_CALL_ARGS deltas are only emitted once the call is
      // open to avoid emitting args with no parent TOOL_CALL_START.
      const prev = ctx.toolCallArgs.get(callId) ?? "";
      ctx.toolCallArgs.set(callId, prev + raw.delta);
      if (ctx.openToolCalls.has(callId)) {
        events.push({
          type: EventType.TOOL_CALL_ARGS,
          toolCallId: callId,
          delta: raw.delta,
        });
      }
      break;
    }
    case "response.reasoning_summary_text.delta":
    case "response.reasoning_text.delta": {
      // Shape: { type, item_id, delta, output_index, sequence_number, ... }
      if (typeof raw.delta !== "string") break;
      const itemId = typeof raw.item_id === "string" ? raw.item_id : "reasoning";
      if (!ctx.reasoningOpen || ctx.currentReasoningId !== itemId) {
        ctx.currentReasoningId = itemId;
        ctx.reasoningOpen = true;
        events.push({
          type: EventType.REASONING_MESSAGE_START,
          messageId: itemId,
          role: "reasoning",
        });
      }
      events.push({
        type: EventType.REASONING_MESSAGE_CONTENT,
        messageId: itemId,
        delta: raw.delta,
      });
      break;
    }
    default:
      // Other raw model events (refusals, output_item.added, etc.) are not
      // mapped in Phase 2; the run_item events carry the higher-level truth.
      break;
  }
  return events;
}

/* ------------------------------ run item events ---------------------------- */

/**
 * Handle a `run_item_stream_event` (identified by `name`).
 *
 * `item` is a `RunItem` subclass. We read its `type` and `rawItem` to extract
 * the fields AG-UI cares about (tool name, call id, args, output).
 */
export function handleRunItemStreamEvent(
  name: string,
  item: any,
  ctx: StreamContext,
): BaseEvent[] {
  const events: BaseEvent[] = [];
  if (!item) return events;

  switch (name) {
    case "message_output_created":
      return handleMessageOutputCreated(item, ctx);

    case "tool_called":
      return handleToolCalled(item, ctx);

    case "tool_output":
      return handleToolOutput(item, ctx);

    case "reasoning_item_created":
      return handleReasoningItemCreated(item, ctx);

    case "tool_approval_requested":
    case "handoff_requested":
    case "handoff_occurred":
    case "tool_search_called":
    case "tool_search_output_created":
      // Surfaced as CUSTOM for now; full HITL pause/resume lands in Phase 5.
      events.push({
        type: EventType.CUSTOM,
        name,
        value: itemToJSON(item),
      });
      return events;

    default:
      return events;
  }
}

function handleMessageOutputCreated(item: any, ctx: StreamContext): BaseEvent[] {
  const events: BaseEvent[] = [];
  // Close any open reasoning block before the assistant message lands.
  if (ctx.reasoningOpen) {
    events.push({
      type: EventType.REASONING_MESSAGE_END,
      messageId: ctx.currentReasoningId!,
    });
    ctx.reasoningOpen = false;
  }
  // Close the streaming text message (text deltas already emitted its content).
  if (ctx.messageOpen) {
    events.push({
      type: EventType.TEXT_MESSAGE_END,
      messageId: ctx.currentMessageId!,
    });
    ctx.messageOpen = false;
  }
  return events;
}

function handleToolCalled(item: any, ctx: StreamContext): BaseEvent[] {
  const events: BaseEvent[] = [];
  const raw = item?.rawItem;
  if (!raw) return events;
  // function_call items carry callId/name/arguments; hosted/computer/etc.
  // calls carry callId too.
  const callId: string | undefined = raw.callId ?? raw.call_id;
  const toolName: string | undefined = raw.name;
  if (!callId) return events;

  if (toolName) ctx.toolCallNames.set(callId, toolName);

  // If args streamed before the item, emit them now as a single ARGS delta.
  const buffered = ctx.toolCallArgs.get(callId);
  ctx.openToolCalls.add(callId);

  events.push({
    type: EventType.TOOL_CALL_START,
    toolCallId: callId,
    toolCallName: toolName ?? "tool",
  });

  if (buffered) {
    events.push({
      type: EventType.TOOL_CALL_ARGS,
      toolCallId: callId,
      delta: buffered,
    });
  } else if (typeof raw.arguments === "string" && raw.arguments.length > 0) {
    // No streamed deltas — emit the full args as one delta.
    events.push({
      type: EventType.TOOL_CALL_ARGS,
      toolCallId: callId,
      delta: raw.arguments,
    });
  }

  events.push({
    type: EventType.TOOL_CALL_END,
    toolCallId: callId,
  });
  return events;
}

function handleToolOutput(item: any, ctx: StreamContext): BaseEvent[] {
  const raw = item?.rawItem;
  const callId: string | undefined = raw?.callId ?? raw?.call_id;
  if (!callId) return [];
  ctx.openToolCalls.delete(callId);
  ctx.toolCallArgs.delete(callId);
  ctx.toolCallNames.delete(callId);
  return [
    {
      type: EventType.TOOL_CALL_RESULT,
      messageId: callId,
      toolCallId: callId,
      content: outputToString(item?.output ?? raw?.output ?? ""),
      role: "tool",
    },
  ];
}

function handleReasoningItemCreated(item: any, ctx: StreamContext): BaseEvent[] {
  // The reasoning summary deltas were already streamed as
  // REASONING_MESSAGE_CONTENT; this item signals the block is complete.
  const events: BaseEvent[] = [];
  if (ctx.reasoningOpen) {
    events.push({
      type: EventType.REASONING_MESSAGE_END,
      messageId: ctx.currentReasoningId!,
    });
    ctx.reasoningOpen = false;
  }
  return events;
}

function itemToJSON(item: any): unknown {
  try {
    return typeof item?.toJSON === "function" ? item.toJSON() : item;
  } catch {
    return undefined;
  }
}

/* --------------------------- agent updated event --------------------------- */

export function handleAgentUpdatedStreamEvent(
  agent: any,
): BaseEvent[] {
  return [
    {
      type: EventType.CUSTOM,
      name: "agent_updated",
      value: { name: agent?.name },
    },
  ];
}

/* ------------------------------ lifecycle helpers -------------------------- */

/**
 * Close any open message / reasoning block at end-of-stream. Returns events
 * to emit (if any) so the stream ends in a valid state.
 */
export function closeOpenBlocks(ctx: StreamContext): BaseEvent[] {
  const events: BaseEvent[] = [];
  if (ctx.messageOpen) {
    events.push({
      type: EventType.TEXT_MESSAGE_END,
      messageId: ctx.currentMessageId!,
    });
    ctx.messageOpen = false;
  }
  if (ctx.reasoningOpen) {
    events.push({
      type: EventType.REASONING_MESSAGE_END,
      messageId: ctx.currentReasoningId!,
    });
    ctx.reasoningOpen = false;
  }
  return events;
}

/**
 * Build the MESSAGES_SNAPSHOT payload from the input messages plus any
 * assistant messages produced during the run.
 *
 * Phase 2 returns the input messages as-is; Phase 4 will merge in generated
 * assistant / tool messages accumulated from the stream. The signature accepts
 * accumulated messages so later phases can extend it without changing callers.
 */
export function buildMessagesSnapshot(
  inputMessages: Message[],
  _accumulated: Message[] = [],
): Message[] {
  return [...inputMessages, ..._accumulated];
}
