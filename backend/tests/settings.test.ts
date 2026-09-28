import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { parse } from 'yaml';
import { ValidationError } from '../src/logic/errors.js';
import { createSettingsService } from '../src/logic/settings.service.js';

let root: string;
let service: ReturnType<typeof createSettingsService>;
let originalQuality: string;
let originalSplits: string;
beforeEach(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), 'settings-'));
  for (const [directory, file] of [
    ['policies', 'quality.yaml'],
    ['splits', 'splits.yaml'],
  ]) {
    await fs.mkdir(path.join(root, directory));
    await fs.copyFile(path.resolve('../app', directory, file), path.join(root, directory, file));
  }
  originalQuality = await fs.readFile(path.join(root, 'policies/quality.yaml'), 'utf8');
  originalSplits = await fs.readFile(path.join(root, 'splits/splits.yaml'), 'utf8');
  service = createSettingsService(root);
});
afterEach(async () => {
  vi.restoreAllMocks();
  await fs.rm(root, { recursive: true, force: true });
});
it('GET exposes only editable configuration', async () => {
  const result = await service.get();
  expect(Object.keys(result)).toEqual(['quality', 'splits']);
  expect(result.quality.min_images_per_class.threshold).toBe(300);
  expect(result.quality).not.toHaveProperty('cross_split_leakage');
  expect(result).not.toHaveProperty('database_url');
  expect(result.splits.seed).toBe(42);
});
it('quality persists, preserves leakage/comments, leaves splits untouched and restores byte-for-byte', async () => {
  const { quality } = await service.get();
  const changed = structuredClone(quality);
  changed.duplicate_similarity_threshold.threshold = 0.98;
  changed.min_images_per_class.action = 'warn';
  await service.saveQuality(changed);
  expect((await createSettingsService(root).get()).quality).toEqual(changed);
  const text = await fs.readFile(path.join(root, 'policies/quality.yaml'), 'utf8');
  expect(parse(text).cross_split_leakage).toEqual(parse(originalQuality).cross_split_leakage);
  expect(text).toContain('# Umbrales');
  expect(await fs.readFile(path.join(root, 'splits/splits.yaml'), 'utf8')).toBe(originalSplits);
  await service.saveQuality(quality);
  expect(await fs.readFile(path.join(root, 'policies/quality.yaml'), 'utf8')).toBe(originalQuality);
});
it.each([
  ['min_images_per_class', 'threshold', -1],
  ['max_imbalance_ratio', 'threshold', Infinity],
  ['min_images_per_class', 'action', 'accept'],
  ['max_small_object_ratio', 'threshold', 1.1],
  ['max_small_object_ratio', 'width_px', 0],
  ['max_small_object_ratio', 'height_px', -1],
  ['duplicate_similarity_threshold', 'threshold', -0.1],
  ['duplicate_similarity_threshold', 'threshold', 1.1],
  ['min_spatial_dispersion', 'threshold', 0.51],
  ['degenerate_boxes', 'threshold', -1],
  ['min_images_per_class', 'extra', 1],
])('rejects quality %s.%s=%s without touching files', async (rule, field, value) => {
  const input = (await service.get()).quality as unknown as Record<string, Record<string, unknown>>;
  input[rule][field] = value;
  await expect(service.saveQuality(input)).rejects.toBeInstanceOf(ValidationError);
  expect(await fs.readFile(path.join(root, 'policies/quality.yaml'), 'utf8')).toBe(originalQuality);
});
it.each(['path', 'database_url', 'cross_split_leakage', '__proto__'])(
  'rejects forbidden key %s and path injection',
  async (key) => {
    const input = { ...(await service.get()).quality, [key]: '../../outside.yaml' };
    await expect(service.saveQuality(input)).rejects.toBeInstanceOf(ValidationError);
    expect(await fs.readFile(path.join(root, 'policies/quality.yaml'), 'utf8')).toBe(
      originalQuality,
    );
  },
);
it('splits persists independently and restores byte-for-byte', async () => {
  const { splits } = await service.get();
  const changed = { train: 0.5, val: 0.3, test: 0.2, seed: 123 };
  expect(await service.saveSplits(changed)).toEqual(changed);
  expect((await createSettingsService(root).get()).splits).toEqual(changed);
  expect(await fs.readFile(path.join(root, 'policies/quality.yaml'), 'utf8')).toBe(originalQuality);
  await service.saveSplits(splits);
  expect(await fs.readFile(path.join(root, 'splits/splits.yaml'), 'utf8')).toBe(originalSplits);
});
it.each([
  { train: 0.5 },
  { train: 0 },
  { val: -1 },
  { test: 1 },
  { seed: 1.5 },
  { seed: Number.MAX_SAFE_INTEGER + 1 },
  { seed: '42' },
  { path: '../../outside.yaml' },
])('rejects invalid splits %j without touching files', async (change) => {
  const input = { ...(await service.get()).splits, ...change };
  await expect(service.saveSplits(input)).rejects.toBeInstanceOf(ValidationError);
  expect(await fs.readFile(path.join(root, 'splits/splits.yaml'), 'utf8')).toBe(originalSplits);
});
it('failed rename preserves original and cleans temp; subsequent saves still work', async () => {
  const input = { ...(await service.get()).splits, seed: 73 };
  vi.spyOn(fs, 'rename').mockRejectedValueOnce(new Error('simulated disk error'));
  await expect(service.saveSplits(input)).rejects.toThrow('simulated disk error');
  expect(await fs.readFile(path.join(root, 'splits/splits.yaml'), 'utf8')).toBe(originalSplits);
  expect(await fs.readdir(path.join(root, 'splits'))).toEqual(['splits.yaml']);
  await service.saveSplits(input);
  expect((await service.get()).splits.seed).toBe(73);
});
it('serializes concurrent writes; last submitted valid write wins', async () => {
  const { splits } = await service.get();
  await Promise.all([10, 20, 30].map((seed) => service.saveSplits({ ...splits, seed })));
  expect((await service.get()).splits.seed).toBe(30);
  expect(await fs.readdir(path.join(root, 'splits'))).toEqual(['splits.yaml']);
});
