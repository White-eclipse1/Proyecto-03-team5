// @vitest-environment node
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, expect, it } from "vitest";
import { readReport } from "../reports-plugin";

let temp: string;
let root: string;
beforeEach(async () => {
  temp = await mkdtemp(path.join(tmpdir(), "p236-"));
  root = path.join(temp, "reports");
  await mkdir(path.join(root, "releases", "v0.1.0"), { recursive: true });
  await writeFile(path.join(root, "quality.json"), "{}");
  await writeFile(path.join(root, "releases", "v0.1.0", "splits.json"), "{}");
  await writeFile(path.join(temp, "private.json"), "secret");
  await symlink(path.join(temp, "private.json"), path.join(root, "escape.json"));
});
afterEach(async () => {
  await rm(temp, { recursive: true, force: true });
});
it.each(["/reports/quality.json", "/reports/releases/v0.1.0/splits.json"])(
  "serves %s",
  async (url) => {
    expect((await readReport(root, url)).toString()).toBe("{}");
  }
);
it.each([
  "/reports/../private.json",
  "/reports/%2e%2e/private.json",
  "/reports/escape.json",
  "/reports/no.json",
  "/reports/%",
  "/reports/a\\b.json",
  "/reports/quality.json/",
  "/other/quality.json",
])("rejects %s", async (url) => {
  await expect(readReport(root, url)).rejects.toThrow();
});

// Invoke the actual Connect HTTP middleware with request/response doubles.
async function requestReport(url: string, method = "GET") {
  const { reportsPlugin } = await import("../reports-plugin");
  type Middleware = import("vite").Connect.NextHandleFunction;
  let handler: Middleware | undefined;
  const server = { middlewares: { use(middleware: Middleware) { handler = middleware; } } };
  const hook = reportsPlugin(root).configureServer;
  if (typeof hook !== "function") throw new Error("Expected configureServer hook");
  hook.call({} as never, server as unknown as import("vite").ViteDevServer);
  const middleware = handler;
  if (!middleware) throw new Error("HTTP middleware was not registered");
  return new Promise<{ status: number; body: string; headers: Record<string, string> }>((resolve) => {
    const headers: Record<string, string> = {};
    const response = {
      statusCode: 200,
      setHeader(name: string, value: string) { headers[name] = value; },
      end(body?: Buffer | string) { resolve({ status: this.statusCode, body: body?.toString() ?? "", headers }); },
    };
    middleware(
      { url, method } as import("node:http").IncomingMessage,
      response as unknown as import("node:http").ServerResponse,
      () => resolve({ status: 418, body: "next middleware", headers })
    );
  });
}

it.each([
  ["/reports/quality.json", "GET", 200],
  ["/reports/releases/v0.1.0/splits.json", "GET", 200],
  ["/reports/quality.json", "HEAD", 200],
  ["/reports/missing.json", "GET", 404],
  ["/reports/../private.json", "GET", 404],
  ["/reports/%2e%2e/private.json", "GET", 404],
  ["/reports/%252e%252e/private.json", "GET", 404],
  ["/reports/quality.json", "POST", 405],
])("HTTP middleware %s %s returns %s", async (url, method, status) => {
  const result = await requestReport(url, method);
  expect(result.status).toBe(status);
  if (status === 200) {
    expect(result.headers["Content-Type"]).toBe("application/json; charset=utf-8");
    if (method === "GET") expect(JSON.parse(result.body)).toEqual({});
    else expect(result.body).toBe("");
  }
});
