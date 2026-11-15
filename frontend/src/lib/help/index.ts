// The help registry behind lib/Info.svelte: one short plain-language entry per ⓘ, keyed
// `<area>.<thing>`, split per domain so pages can grow independently. Keep an entry to one or
// two sentences (help.test.ts caps the length) — the ⓘ explains, the spec documents.
import { chrome } from './chrome';
import { evolution } from './evolution';
import { latent } from './latent';
import { ops } from './ops';
import { trading } from './trading';

export const HELP: Record<string, string> = { ...chrome, ...ops, ...latent, ...evolution, ...trading };

/** The text for a key; '' hides the ⓘ (help.test.ts fails on keys used in pages but missing here). */
export function help(k: string): string {
	return HELP[k] ?? '';
}
