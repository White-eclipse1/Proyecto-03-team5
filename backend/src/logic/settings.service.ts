import { createSettingsRepository } from '../data/repositories/settings.repository.js';
import { ValidationError } from './errors.js';
import {
  qualityFileSchema,
  qualitySettingsSchema,
  splitSettingsSchema,
} from './settings.validation.js';

export function createSettingsService(root: string) {
  const repository = createSettingsRepository(root);
  return {
    async get() {
      const quality = qualityFileSchema.parse(await repository.read('quality'));
      const { cross_split_leakage: _preserved, ...editable } = quality;
      return {
        quality: editable,
        splits: splitSettingsSchema.parse(await repository.read('splits')),
      };
    },
    async saveQuality(input: unknown) {
      const parsed = qualitySettingsSchema.safeParse(input);
      if (!parsed.success)
        throw new ValidationError(
          parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; '),
        );
      await repository.update('quality', (current) => {
        qualityFileSchema.parse(current);
        return parsed.data;
      });
      return parsed.data;
    },
    async saveSplits(input: unknown) {
      const parsed = splitSettingsSchema.safeParse(input);
      if (!parsed.success)
        throw new ValidationError(
          parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; '),
        );
      await repository.update('splits', (current) => {
        splitSettingsSchema.parse(current);
        return parsed.data;
      });
      return parsed.data;
    },
  };
}
