import { randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseDocument } from 'yaml';

type Section = 'quality' | 'splits';

/** Fixed destinations supplied by server wiring, never by an HTTP request. */
export function createSettingsRepository(root: string) {
  const files = {
    quality: path.join(root, 'policies', 'quality.yaml'),
    splits: path.join(root, 'splits', 'splits.yaml'),
  };
  let pending: Promise<unknown> = Promise.resolve();

  async function readDocument(section: Section) {
    const text = await fs.readFile(files[section], 'utf8');
    const document = parseDocument(text, { uniqueKeys: true });
    if (document.errors.length) throw new Error('Configuración YAML inválida.');
    return document;
  }

  return {
    async read(section: Section): Promise<unknown> {
      return (await readDocument(section)).toJS({ maxAliasCount: 0 });
    },
    update(section: Section, transform: (current: unknown) => Record<string, unknown>) {
      const operation = pending.then(async () => {
        const document = await readDocument(section);
        const values = transform(document.toJS({ maxAliasCount: 0 }));
        // Update scalars in place to preserve comments and non-editable fields.
        for (const [key, value] of Object.entries(values)) {
          if (typeof value === 'object' && value !== null) {
            for (const [field, scalar] of Object.entries(value)) {
              document.setIn([key, field], scalar);
            }
          } else {
            document.set(key, value);
          }
        }
        const destination = files[section];
        const temporary = path.join(path.dirname(destination), `.settings-${randomUUID()}.tmp`);
        try {
          const mode = (await fs.stat(destination)).mode & 0o777;
          const handle = await fs.open(temporary, 'wx', mode);
          try {
            await handle.writeFile(document.toString(), 'utf8');
            await handle.sync();
          } finally {
            await handle.close();
          }
          await fs.rename(temporary, destination);
        } finally {
          await fs.rm(temporary, { force: true });
        }
        return values;
      });
      // A failed write must not poison the queue.
      pending = operation.catch(() => undefined);
      return operation;
    },
  };
}
