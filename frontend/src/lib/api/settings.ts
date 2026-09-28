import { z } from "zod";

const thresholdRule = z
  .object({
    threshold: z.number().finite().nonnegative(),
    action: z.enum(["warn", "fail"]),
  })
  .strict();

// Mirrors the domains of the Python analyzer configs, not just ThresholdRule.
export const qualitySettingsSchema = z
  .object({
    min_images_per_class: thresholdRule,
    max_imbalance_ratio: thresholdRule,
    max_small_object_ratio: thresholdRule
      .extend({
        threshold: z.number().finite().min(0).max(1),
        width_px: z.number().finite().positive(),
        height_px: z.number().finite().positive(),
      })
      .strict(),
    degenerate_boxes: thresholdRule,
    duplicate_similarity_threshold: thresholdRule
      .extend({
        threshold: z.number().finite().min(0).max(1),
      })
      .strict(),
    min_spatial_dispersion: thresholdRule
      .extend({
        threshold: z.number().finite().min(0).max(0.5),
      })
      .strict(),
  })
  .strict();

const ratio = z.number().finite().gt(0).lt(1);
export const splitSettingsSchema = z
  .object({
    train: ratio,
    val: ratio,
    test: ratio,
    seed: z.number().int().min(Number.MIN_SAFE_INTEGER).max(Number.MAX_SAFE_INTEGER),
  })
  .strict()
  .refine(
    (value) => Math.abs(value.train + value.val + value.test - 1) <= 1e-6,
    "train + validation + test debe sumar 1 (100%)."
  );

export const pipelineSettingsSchema = z
  .object({
    quality: qualitySettingsSchema,
    splits: splitSettingsSchema,
  })
  .strict();
export type QualitySettings = z.infer<typeof qualitySettingsSchema>;
export type SplitSettings = z.infer<typeof splitSettingsSchema>;

import { apiRequest, jsonBody } from "./client";

export async function getSettings(signal?: AbortSignal) {
  return apiRequest("/settings", pipelineSettingsSchema, { signal });
}
export async function saveQualitySettings(value: QualitySettings) {
  return apiRequest("/settings/quality", qualitySettingsSchema, {
    method: "PUT",
    ...jsonBody(value),
  });
}
export async function saveSplitSettings(value: SplitSettings) {
  return apiRequest("/settings/splits", splitSettingsSchema, { method: "PUT", ...jsonBody(value) });
}
