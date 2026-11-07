// UI preferences in localStorage, wrapped in try/catch (spec 51 §5). Never holds data, only view state.
const PREFIX = 'fa.ui.';

export function loadPref<T>(key: string, fallback: T): T {
	try {
		const raw = localStorage.getItem(PREFIX + key);
		return raw === null ? fallback : (JSON.parse(raw) as T);
	} catch {
		return fallback;
	}
}

export function savePref(key: string, value: unknown): void {
	try {
		localStorage.setItem(PREFIX + key, JSON.stringify(value));
	} catch {
		// private mode / storage disabled: preferences just don't persist
	}
}
