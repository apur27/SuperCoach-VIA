/** Browser-side validation: loads only the validator for the requested resource kind. */
import type { ResourceKind, ResourceTypes } from './contracts.generated';
import { VALIDATOR_LOADERS } from './validators/index.generated';

export async function validateAsync<K extends ResourceKind>(kind: K, data: unknown): Promise<{ ok: true; value: ResourceTypes[K] } | { ok: false; error: string }> {
  const fn = await VALIDATOR_LOADERS[kind]();
  if (fn(data)) return { ok: true, value: data };
  const e = fn.errors?.[0];
  return { ok: false, error: e ? `${e.instancePath || '/'} ${e.message ?? e.keyword}` : 'invalid' };
}
