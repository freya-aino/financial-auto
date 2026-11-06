import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// SvelteKit 3: config lives in the plugin options (no svelte.config.js). D-015.
const api = 'http://127.0.0.1:7700';

export default defineConfig({
	plugins: [
		sveltekit({
			adapter: adapter({ fallback: '200.html' })
		})
	],
	server: {
		port: 5173,
		host: '127.0.0.1',
		proxy: {
			'/api': api,
			'/ws': { target: api.replace('http', 'ws'), ws: true }
		}
	}
});
