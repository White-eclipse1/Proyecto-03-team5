import { readFile, realpath } from "node:fs/promises";
import path from "node:path";
import type { Plugin } from "vite";

/** Serve only JSON reports, including release directories, never symlink escapes. */
export async function readReport(root: string, url: string): Promise<Buffer> {
  const pathname = decodeURIComponent(url.split("?")[0] ?? "");
  if (!pathname.startsWith("/reports/")) throw new Error("Invalid report path");
  const parts = pathname.slice("/reports/".length).split("/");
  if (parts.some((part) => !/^[A-Za-z0-9_-][A-Za-z0-9._-]*$/.test(part))) {
    throw new Error("Invalid report path");
  }
  if (!parts.at(-1)?.endsWith(".json")) throw new Error("Only JSON reports are public");
  const base = await realpath(root);
  const file = await realpath(path.join(base, ...parts));
  if (!file.startsWith(`${base}${path.sep}`)) throw new Error("Report outside root");
  return readFile(file);
}

export function reportsPlugin(root: string): Plugin {
  return {
    name: "dataset-reports",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith("/reports/")) return next();
        if (req.method !== "GET" && req.method !== "HEAD") {
          res.statusCode = 405;
          res.end();
          return;
        }
        void readReport(root, req.url).then(
          (body) => {
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.setHeader("Cache-Control", "no-store");
            res.end(req.method === "HEAD" ? undefined : body);
          },
          () => {
            res.statusCode = 404;
            res.end("Report not found");
          }
        );
      });
    },
  };
}
