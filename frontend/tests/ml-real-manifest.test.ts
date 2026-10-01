import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { trainingManifestSchema } from "../src/ml/schemas";

describe("OPS-02 real manifest contract", () => {
  it("parses the committed v0.1.1 manifest with the frontend schema", () => {
    const manifestPath = resolve(
      process.cwd(),
      "../reports/releases/v0.1.1/manifest.json"
    );

    const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
    const result = trainingManifestSchema.safeParse(manifest);

    expect(result.success).toBe(true);
  });
});
