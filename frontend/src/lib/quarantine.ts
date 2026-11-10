// Quarantine resolve form (D-054): activate the provisional instrument with identifiers, merge
// it into an existing instrument, or ignore. Validation mirrors what the API accepts.
export type ResolveMode = 'activate' | 'merge' | 'ignore';

export interface ResolveForm {
	mode: ResolveMode;
	isin: string;
	figi: string;
	asset_class: string;
	kind: string;
	name: string;
	base_ccy: string;
	quote_ccy: string;
	merge_into: string;
}

export const ASSET_CLASSES = ['equity', 'etf', 'fx', 'crypto', 'index', 'commodity', 'bond', 'option', 'future', 'cfd', 'news', 'rates'];
export const INSTRUMENT_KINDS = ['equity', 'etf', 'fx_pair', 'crypto_pair', 'index', 'future', 'option', 'bond', 'cfd', 'other'];

export const emptyResolveForm = (): ResolveForm => ({
	mode: 'activate',
	isin: '',
	figi: '',
	asset_class: '',
	kind: '',
	name: '',
	base_ccy: '',
	quote_ccy: '',
	merge_into: ''
});

const ISIN = /^[A-Z]{2}[A-Z0-9]{9}\d$/;
const FIGI = /^BBG[A-Z0-9]{9}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** ISIN mod-97 style check digit (Luhn over the letter-expanded string). */
export function isinChecksumOk(isin: string): boolean {
	if (!ISIN.test(isin)) return false;
	const digits = [...isin].map((c) => (/\d/.test(c) ? c : String(c.charCodeAt(0) - 55))).join('');
	let sum = 0;
	let dbl = false;
	for (let i = digits.length - 1; i >= 0; i--) {
		let d = Number(digits[i]);
		if (dbl) {
			d *= 2;
			if (d > 9) d -= 9;
		}
		sum += d;
		dbl = !dbl;
	}
	return sum % 10 === 0;
}

export interface Validation {
	errors: string[];
	/** Request body for `POST /api/instruments/quarantine/{id}/resolve`, when valid. */
	body: Record<string, unknown> | null;
}

export function validateResolve(f: ResolveForm): Validation {
	const errors: string[] = [];
	const t = (s: string) => s.trim();
	if (f.mode === 'ignore') return { errors, body: { ignore: true } };
	if (f.mode === 'merge') {
		if (!UUID.test(t(f.merge_into))) errors.push('merge target must be an instrument UUID');
		return { errors, body: errors.length ? null : { instrument_id: t(f.merge_into).toLowerCase() } };
	}
	const isin = t(f.isin).toUpperCase();
	const figi = t(f.figi).toUpperCase();
	if (isin && !isinChecksumOk(isin)) errors.push('ISIN must be 12 characters (2 letters, 9 alphanumerics, check digit) with a valid check digit');
	if (figi && !FIGI.test(figi)) errors.push('FIGI must be BBG followed by 9 alphanumerics');
	if (!t(f.asset_class)) errors.push('asset class is required to activate');
	if (!t(f.kind)) errors.push('kind is required to activate');
	if (f.base_ccy && !/^[A-Z]{3}$/.test(t(f.base_ccy).toUpperCase())) errors.push('base currency must be a 3-letter code');
	if (f.quote_ccy && !/^[A-Z]{3}$/.test(t(f.quote_ccy).toUpperCase())) errors.push('quote currency must be a 3-letter code');
	if (errors.length) return { errors, body: null };
	const body: Record<string, unknown> = { asset_class: t(f.asset_class), kind: t(f.kind) };
	if (isin) body.isin = isin;
	if (figi) body.figi = figi;
	if (t(f.name)) body.name = t(f.name);
	if (t(f.base_ccy)) body.base_ccy = t(f.base_ccy).toUpperCase();
	if (t(f.quote_ccy)) body.quote_ccy = t(f.quote_ccy).toUpperCase();
	return { errors, body };
}
