// Reproduces #109638: a session that spawns a subagent, performs more tool
// calls, then calls sessions_yield must still receive the child's completion
// auto-announce as a new turn instead of deadlocking on transcript state.
import { mkdtemp, rm } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../src/config/types.openclaw.js";
import { GatewayChatClient } from "../src/tui/gateway-chat.js";
import {
  createOpenClawTestInstance,
  type OpenClawTestInstance,
} from "./helpers/openclaw-test-instance.js";

type YieldScenario = "work-between-spawn-and-yield" | "immediate-yield" | "compaction-before-wake";

type MockModelRequest = {
  kind: "parent" | "child";
  branch: "spawn-toolcall" | "work-toolcall" | "yield-toolcall" | "post-yield-text" | "child-text";
  bodyText: string;
};

type MockModelServer = {
  baseUrl: string;
  requests: MockModelRequest[];
  announceRequests: MockModelRequest[];
  stop: () => Promise<void>;
};

const TEST_TIMEOUT_MS = 300_000;
const ANNOUNCE_WAIT_OPTS = { timeout: 180_000, interval: 100 } as const;

const PARENT_KICKOFF_MARKER = "OPENCLAW_E2E_YIELD_PARENT_KICKOFF";
const CHILD_RESULT_MARKER = "OPENCLAW_E2E_YIELD_CHILD_RESULT";
const ANNOUNCE_ACK_MARKER = "OPENCLAW_E2E_YIELD_ANNOUNCE_ACK";

const instances: OpenClawTestInstance[] = [];
const cleanupDirs: string[] = [];
const modelServers: MockModelServer[] = [];

afterEach(async () => {
  await Promise.all(instances.splice(0).map((instance) => instance.cleanup()));
  await Promise.all(modelServers.splice(0).map((server) => server.stop()));
  await Promise.all(cleanupDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function readJsonRequest(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const body = Buffer.concat(chunks).toString("utf8");
  return body ? (JSON.parse(body) as Record<string, unknown>) : {};
}

function writeSse(res: ServerResponse, events: Array<Record<string, unknown>>): void {
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-store",
    connection: "keep-alive",
  });
  res.end(
    `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`,
  );
}

let mockItemCounter = 0;

function textResponseEvents(text: string): Array<Record<string, unknown>> {
  mockItemCounter += 1;
  const itemId = `msg_yield_e2e_${mockItemCounter}`;
  const message = {
    type: "message",
    id: itemId,
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  };
  return [
    {
      type: "response.output_item.added",
      output_index: 0,
      item: { ...message, status: "in_progress", content: [] },
    },
    {
      type: "response.output_text.delta",
      item_id: itemId,
      output_index: 0,
      content_index: 0,
      delta: text,
    },
    { type: "response.output_text.done", item_id: itemId, output_index: 0, content_index: 0, text },
    { type: "response.output_item.done", output_index: 0, item: message },
    {
      type: "response.completed",
      response: {
        id: `resp_yield_e2e_${mockItemCounter}`,
        status: "completed",
        output: [message],
        usage: { input_tokens: 11, output_tokens: 7, total_tokens: 18 },
      },
    },
  ];
}

function toolCallResponseEvents(
  name: string,
  args: Record<string, unknown>,
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number },
): Array<Record<string, unknown>> {
  mockItemCounter += 1;
  const serialized = JSON.stringify(args);
  const itemId = `fc_yield_e2e_${mockItemCounter}`;
  const item = {
    type: "function_call",
    id: itemId,
    call_id: `call_yield_e2e_${mockItemCounter}`,
    name,
    arguments: serialized,
  };
  return [
    {
      type: "response.output_item.added",
      output_index: 0,
      item: { ...item, arguments: "" },
    },
    {
      type: "response.function_call_arguments.delta",
      item_id: itemId,
      output_index: 0,
      delta: serialized,
    },
    { type: "response.output_item.done", output_index: 0, item },
    {
      type: "response.completed",
      response: {
        id: `resp_yield_e2e_${mockItemCounter}`,
        status: "completed",
        output: [item],
        usage: usage ?? { input_tokens: 64, output_tokens: 16, total_tokens: 80 },
      },
    },
  ];
}

function listFunctionCallNames(body: Record<string, unknown>): string[] {
  const input = Array.isArray(body.input) ? body.input : [];
  return input
    .filter(
      (item): item is { type: string; name?: unknown } =>
        Boolean(item) &&
        typeof item === "object" &&
        (item as { type?: unknown }).type === "function_call",
    )
    .map((item) => String(item.name ?? ""));
}

async function startMockModelServer(scenario: YieldScenario): Promise<MockModelServer> {
  const requests: MockModelRequest[] = [];
  const announceRequests: MockModelRequest[] = [];
  const server = createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/v1/models") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ data: [{ id: "yield-e2e", object: "model" }] }));
        return;
      }
      if (req.method !== "POST" || url.pathname !== "/v1/responses") {
        res.writeHead(404).end();
        return;
      }
      const body = await readJsonRequest(req);
      const bodyText = JSON.stringify(body);
      const isParent = bodyText.includes(PARENT_KICKOFF_MARKER);
      if (!isParent) {
        // Child session: complete immediately with the marker result.
        requests.push({ kind: "child", branch: "child-text", bodyText });
        writeSse(res, textResponseEvents(CHILD_RESULT_MARKER));
        return;
      }
      const toolCallNames = listFunctionCallNames(body);
      if (!toolCallNames.includes("sessions_spawn")) {
        requests.push({ kind: "parent", branch: "spawn-toolcall", bodyText });
        writeSse(
          res,
          toolCallResponseEvents("sessions_spawn", {
            task: `Reply with exactly ${CHILD_RESULT_MARKER} and nothing else.`,
            label: "yield-e2e-child",
          }),
        );
        return;
      }
      if (scenario !== "immediate-yield" && !toolCallNames.includes("session_status")) {
        // Intervening tool work between spawn and yield (the failing shape in
        // the report; immediate yield is the control).
        requests.push({ kind: "parent", branch: "work-toolcall", bodyText });
        writeSse(res, toolCallResponseEvents("session_status", {}));
        return;
      }
      if (!toolCallNames.includes("sessions_yield")) {
        requests.push({ kind: "parent", branch: "yield-toolcall", bodyText });
        // For the compaction scenario, report near-window usage on the yield
        // turn so the wake prompt crosses the auto-compaction threshold
        // (contextWindow 128000 minus default reserveTokens 16384).
        const usage =
          scenario === "compaction-before-wake"
            ? { input_tokens: 120_000, output_tokens: 200, total_tokens: 120_200 }
            : undefined;
        writeSse(
          res,
          toolCallResponseEvents(
            "sessions_yield",
            { message: "Waiting for the spawned child result." },
            usage,
          ),
        );
        return;
      }
      // A parent request after sessions_yield that carries the child result is
      // the child-completion wake turn.
      requests.push({ kind: "parent", branch: "post-yield-text", bodyText });
      if (bodyText.includes(CHILD_RESULT_MARKER)) {
        announceRequests.push({ kind: "parent", branch: "post-yield-text", bodyText });
      }
      writeSse(res, textResponseEvents(ANNOUNCE_ACK_MARKER));
    })().catch((error: unknown) => {
      if (!res.headersSent) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: String(error) } }));
        return;
      }
      res.destroy(error instanceof Error ? error : new Error(String(error)));
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("mock model server did not bind");
  }
  let stopped = false;
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    requests,
    announceRequests,
    stop: async () => {
      if (stopped) {
        return;
      }
      stopped = true;
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
        server.closeAllConnections();
      });
    },
  };
}

async function runYieldAnnounceScenario(scenario: YieldScenario): Promise<{
  server: MockModelServer;
  instance: OpenClawTestInstance;
}> {
  const fixtureDir = await mkdtemp(path.join(tmpdir(), "openclaw-yield-announce-"));
  cleanupDirs.push(fixtureDir);
  const server = await startMockModelServer(scenario);
  modelServers.push(server);
  const modelRef = "yield-e2e/yield-e2e";
  const config = {
    agents: {
      defaults: {
        workspace: path.join(fixtureDir, "workspace"),
        model: { primary: modelRef },
        models: { [modelRef]: { agentRuntime: { id: "openclaw" } } },
        skills: [],
        skipBootstrap: true,
      },
      list: [{ id: "main", default: true, model: { primary: modelRef }, skills: [] }],
    },
    tools: { profile: "coding" },
    models: {
      mode: "replace",
      providers: {
        "yield-e2e": {
          baseUrl: `${server.baseUrl}/v1`,
          apiKey: "secret-token",
          api: "openai-responses",
          request: { allowPrivateNetwork: true },
          models: [
            {
              id: "yield-e2e",
              name: "yield-e2e",
              api: "openai-responses",
              reasoning: false,
              input: ["text"],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 128_000,
              maxTokens: 4_096,
            },
          ],
        },
      },
    },
  } satisfies OpenClawConfig;
  const instance = await createOpenClawTestInstance({
    name: `yield-announce-${scenario}`,
    gatewayToken: "secret-token",
    config,
    env: {
      OPENCLAW_SKIP_PROVIDERS: undefined,
      OPENCLAW_TEST_FAST: "1",
    },
  });
  instances.push(instance);
  await instance.startGateway();

  const client = new GatewayChatClient({
    url: instance.url,
    token: "secret-token",
    allowInsecureLocalOperatorUi: false,
  });
  client.start();
  await client.waitForReady();
  const sessionKey = `agent:main:yield-announce-${scenario}`;
  try {
    const kickoff = await client.sendChat({
      sessionKey,
      message: PARENT_KICKOFF_MARKER,
      runId: `yield-announce-${scenario}`,
    });
    expect(kickoff.status).toBe("started");

    // The parent must be issued the sessions_yield tool call and the child
    // session must run.
    await vi.waitFor(() => {
      expect(server.requests.some((request) => request.branch === "yield-toolcall")).toBe(true);
      expect(server.requests.some((request) => request.kind === "child")).toBe(true);
    }, ANNOUNCE_WAIT_OPTS);

    // Child completion auto-announce must wake the yielded parent with a new
    // turn. With #109638 unfixed this never happens: delivery dies with
    // "Cannot continue from message role: assistant".
    try {
      await vi.waitFor(() => {
        expect(server.announceRequests.length).toBeGreaterThan(0);
      }, ANNOUNCE_WAIT_OPTS);
    } catch (error) {
      const logs = instance.logs();
      const roleErrors = logs
        .split("\n")
        .filter((line) => line.includes("Cannot continue from message role"))
        .slice(0, 5);
      throw new Error(
        `subagent completion announce never woke the yielded parent (scenario=${scenario}).\n` +
          `parent/child model requests: ${JSON.stringify(
            server.requests.map((request) => request.branch),
          )}\n` +
          (roleErrors.length > 0
            ? `gateway role-continuity errors:\n${roleErrors.join("\n")}`
            : "no role-continuity errors found in gateway logs"),
        { cause: error },
      );
    }
    const announceBody = server.announceRequests[0]?.bodyText ?? "";
    expect(announceBody).toContain(CHILD_RESULT_MARKER);
    // The replayed transcript must contain the executed sessions_yield tool
    // result, proving the wake turn continued from a genuinely yielded state.
    expect(announceBody).toContain("yielded");
  } finally {
    await client.abortChat({ sessionKey }).catch(() => undefined);
    client.stop();
  }
  return { server, instance };
}

describe("sessions_yield subagent completion announce", () => {
  it(
    "wakes the yielded parent when tool work happened between spawn and yield (#109638)",
    async () => {
      await runYieldAnnounceScenario("work-between-spawn-and-yield");
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "wakes the yielded parent after spawn followed by an immediate yield",
    async () => {
      await runYieldAnnounceScenario("immediate-yield");
    },
    TEST_TIMEOUT_MS,
  );

  it(
    "wakes the yielded parent when the wake prompt triggers threshold compaction",
    async () => {
      await runYieldAnnounceScenario("compaction-before-wake");
    },
    TEST_TIMEOUT_MS,
  );
});
