<script lang="ts">
	// Interactive point cloud on a Canvas 2D (spec 51 §6.3, Q1: no chart library). 2-D: drag to
	// pan, wheel to zoom about the cursor. 3-D: drag to orbit, shift-drag (or right-drag) to pan,
	// wheel to zoom. Double-click resets the view. Points are written into an ImageData buffer
	// grouped by colour (no per-point canvas call), overlays (axes, live markers, hover) are vector;
	// redraws are coalesced per animation frame.
	import { onMount } from 'svelte';
	import { HOME, HOME_2D, groupByColor, packRgba, nearest, orbit, pan, projectAll, toScreen, zoomAt, type Camera, type Cloud, type Coloring, type Marker } from './pointcloud';

	interface Props {
		cloud: Cloud;
		coloring: Coloring;
		/** Points with this flag set are drawn on top at full opacity; others are dimmed. */
		emphasis?: Uint8Array | null;
		markers?: Marker[];
		/** Tooltip text of point `i`. */
		describe: (i: number) => string;
		onpick?: (i: number) => void;
		height?: number;
		axes?: string[];
	}
	let { cloud, coloring, emphasis = null, markers = [], describe, onpick, height = 420, axes = ['PC1', 'PC2', 'PC3'] }: Props = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let w = $state(600);
	const h = $derived(height);
	let cam = $state<Camera>({ ...HOME_2D });
	let hover = $state<{ i: number; x: number; y: number } | null>(null);
	let drawMs = $state(0);
	let screen = new Float32Array(0);
	let frame = 0;
	let img: ImageData | null = null;
	let pixels = new Uint32Array(0);

	const is3d = $derived(cloud.dims === 3);
	/** Point indices grouped by palette slot (one fill per colour, no per-frame scan per colour). */
	const groups = $derived(groupByColor(coloring));

	export function reset() {
		cam = { ...(is3d ? HOME : HOME_2D) };
	}

	// a new cloud of a different dimensionality starts from its home view
	let lastDims = 0;
	$effect(() => {
		if (cloud.dims !== lastDims) {
			lastDims = cloud.dims;
			reset();
		}
	});

	function schedule() {
		if (frame || typeof requestAnimationFrame === 'undefined') return;
		frame = requestAnimationFrame(() => {
			frame = 0;
			draw();
		});
	}

	function cssVar(name: string, fallback: string): string {
		if (!canvas) return fallback;
		return getComputedStyle(canvas).getPropertyValue(name).trim() || fallback;
	}

	function draw() {
		if (!canvas) return;
		const t0 = performance.now();
		const dpr = window.devicePixelRatio || 1;
		if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
			canvas.width = Math.round(w * dpr);
			canvas.height = Math.round(h * dpr);
		}
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		if (screen.length < cloud.n * 2) screen = new Float32Array(cloud.n * 2);
		projectAll(cloud, cam, w, h, screen);

		// points: written straight into an ImageData buffer (one Uint32 store per pixel), which
		// stays fast without GPU canvas acceleration; vector overlays are drawn on top
		const W = canvas.width;
		const H = canvas.height;
		if (!img || img.width !== W || img.height !== H) {
			img = ctx.createImageData(W, H);
			pixels = new Uint32Array(img.data.buffer);
		}
		pixels.fill(0);
		const sz = Math.max(1, Math.round((cloud.n > 20_000 ? 1.6 : cloud.n > 5_000 ? 2.2 : 3) * dpr));
		const half = sz / 2;
		// props are getters in Svelte 5: read them once, not per point
		const { palette } = coloring;
		const { order, start } = groups;
		const n = cloud.n;
		const em = emphasis;
		const scr = screen;
		const buf = pixels;
		const passes: [boolean, number][] = em ? [[false, 0.22], [true, 1]] : [[false, 0.9]];
		for (const [emph, alpha] of passes) {
			for (let c = 0; c < palette.length; c++) {
				const px = packRgba(palette[c], alpha);
				for (let k = start[c]; k < start[c + 1]; k++) {
					const i = order[k];
					if (i >= n) continue;
					if (em && (em[i] === 1) !== emph) continue;
					const x0 = Math.round(scr[i * 2] * dpr - half);
					const y0 = Math.round(scr[i * 2 + 1] * dpr - half);
					if (x0 < 0 || y0 < 0 || x0 + sz > W || y0 + sz > H) continue;
					for (let dy = 0; dy < sz; dy++) {
						const row = (y0 + dy) * W + x0;
						for (let dx = 0; dx < sz; dx++) buf[row + dx] = px;
					}
				}
			}
		}
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.putImageData(img, 0, 0);
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		drawAxes(ctx);

		// live markers: trail + ring + label
		const ink = cssVar('--text', '#d6dde6');
		const bg = cssVar('--bg-raised', '#151b23');
		ctx.font = '11px ui-monospace, monospace';
		for (const m of markers) {
			const [x, y] = toScreen(cam, w, h, ...m.pos);
			if (m.trail?.length) {
				ctx.strokeStyle = m.color;
				ctx.globalAlpha = 0.6;
				ctx.lineWidth = 1.5;
				ctx.setLineDash([3, 3]);
				ctx.beginPath();
				m.trail.forEach((p, k) => {
					const [tx, ty] = toScreen(cam, w, h, ...p);
					if (k === 0) ctx.moveTo(tx, ty);
					else ctx.lineTo(tx, ty);
				});
				ctx.lineTo(x, y);
				ctx.stroke();
				ctx.setLineDash([]);
				ctx.globalAlpha = 1;
			}
			ctx.beginPath();
			ctx.arc(x, y, 7, 0, Math.PI * 2);
			ctx.fillStyle = bg;
			ctx.fill();
			ctx.lineWidth = 3;
			ctx.strokeStyle = m.color;
			ctx.stroke();
			ctx.fillStyle = ink;
			ctx.fillText(m.label, x + 10, y + 4);
		}

		if (hover && hover.i < cloud.n) {
			const x = screen[hover.i * 2];
			const y = screen[hover.i * 2 + 1];
			ctx.beginPath();
			ctx.arc(x, y, 5, 0, Math.PI * 2);
			ctx.lineWidth = 2;
			ctx.strokeStyle = ink;
			ctx.stroke();
		}
		drawMs = performance.now() - t0;
	}

	function drawAxes(ctx: CanvasRenderingContext2D) {
		const dim = cssVar('--text-dim', '#8592a3');
		const border = cssVar('--border', '#262f3a');
		ctx.lineWidth = 1;
		ctx.font = '10px ui-monospace, monospace';
		const [ox, oy] = toScreen(cam, w, h, 0, 0, 0);
		const ends: [number, number, number][] = is3d ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : [[1, 0, 0], [0, 1, 0]];
		ends.forEach((e, k) => {
			const [x, y] = toScreen(cam, w, h, ...e);
			const [nx, ny] = toScreen(cam, w, h, -e[0], -e[1], -e[2]);
			ctx.strokeStyle = border;
			ctx.beginPath();
			ctx.moveTo(nx, ny);
			ctx.lineTo(x, y);
			ctx.stroke();
			ctx.fillStyle = dim;
			ctx.fillText(axes[k] ?? '', x + 4, y - 4);
		});
		ctx.fillStyle = dim;
		ctx.fillRect(ox - 1, oy - 1, 2, 2);
	}

	// redraw on any input change
	$effect(() => {
		void cloud;
		void coloring;
		void emphasis;
		void markers;
		void cam;
		void w;
		void h;
		void hover;
		schedule();
	});

	// ---------------------------------------------------------------- input
	let drag: { x: number; y: number; moved: boolean; mode: 'pan' | 'orbit' } | null = null;

	function local(ev: PointerEvent | WheelEvent | MouseEvent): [number, number] {
		const r = canvas!.getBoundingClientRect();
		return [ev.clientX - r.left, ev.clientY - r.top];
	}

	function down(ev: PointerEvent) {
		canvas?.setPointerCapture(ev.pointerId);
		const [x, y] = local(ev);
		drag = { x, y, moved: false, mode: !is3d || ev.shiftKey || ev.button === 2 ? 'pan' : 'orbit' };
	}

	function move(ev: PointerEvent) {
		const [x, y] = local(ev);
		if (drag) {
			const dx = x - drag.x;
			const dy = y - drag.y;
			if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
			cam = drag.mode === 'pan' ? pan(cam, dx, dy) : orbit(cam, dx, dy);
			drag.x = x;
			drag.y = y;
			hover = null;
			return;
		}
		const i = nearest(screen, cloud.n, x, y, 6);
		hover = i < 0 ? null : { i, x, y };
	}

	function up(ev: PointerEvent) {
		const d = drag;
		drag = null;
		canvas?.releasePointerCapture(ev.pointerId);
		if (d && !d.moved && onpick) {
			const [x, y] = local(ev);
			const i = nearest(screen, cloud.n, x, y, 8);
			if (i >= 0) onpick(i);
		}
	}

	function wheel(ev: WheelEvent) {
		ev.preventDefault();
		const [x, y] = local(ev);
		cam = zoomAt(cam, Math.exp(-ev.deltaY * 0.0015), x, y, w, h);
	}

	function key(ev: KeyboardEvent) {
		const step = 40;
		if (ev.key === '+' || ev.key === '=') cam = zoomAt(cam, 1.25, w / 2, h / 2, w, h);
		else if (ev.key === '-') cam = zoomAt(cam, 0.8, w / 2, h / 2, w, h);
		else if (ev.key === 'ArrowLeft') cam = is3d && !ev.shiftKey ? orbit(cam, -step / 2, 0) : pan(cam, step, 0);
		else if (ev.key === 'ArrowRight') cam = is3d && !ev.shiftKey ? orbit(cam, step / 2, 0) : pan(cam, -step, 0);
		else if (ev.key === 'ArrowUp') cam = is3d && !ev.shiftKey ? orbit(cam, 0, -step / 2) : pan(cam, 0, step);
		else if (ev.key === 'ArrowDown') cam = is3d && !ev.shiftKey ? orbit(cam, 0, step / 2) : pan(cam, 0, -step);
		else if (ev.key === '0') reset();
		else return;
		ev.preventDefault();
	}

	let box: HTMLDivElement | undefined = $state();
	onMount(() => {
		const ro = new ResizeObserver(([e]) => {
			w = Math.max(200, Math.round(e.contentRect.width));
		});
		if (box) ro.observe(box);
		// wheel must be non-passive to stop the page scrolling while zooming
		canvas?.addEventListener('wheel', wheel, { passive: false });
		return () => {
			ro.disconnect();
			canvas?.removeEventListener('wheel', wheel);
			if (frame) cancelAnimationFrame(frame);
		};
	});
</script>

<!-- a keyboard-operable canvas widget (arrows / +/- / 0); role="application" is the honest role -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
<div
	class="cloud"
	bind:this={box}
	tabindex="0"
	role="application"
	aria-label="{cloud.n} latent points, {cloud.dims}-D; drag to {is3d ? 'orbit (shift-drag pans)' : 'pan'}, wheel or +/- to zoom, arrows to {is3d ? 'orbit' : 'pan'}, 0 to reset"
	onkeydown={key}
>
	<canvas
		bind:this={canvas}
		style="width: {w}px; height: {h}px"
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onpointerleave={() => (hover = null)}
		ondblclick={reset}
		oncontextmenu={(e) => e.preventDefault()}
	></canvas>
	{#if hover}
		<div class="tip mono" style="left: {Math.min(hover.x + 12, w - 260)}px; top: {hover.y + 12}px">{describe(hover.i)}</div>
	{/if}
	<div class="hud mono dim">
		<span>{cloud.n.toLocaleString()} pts · zoom {cam.zoom.toFixed(1)}× · {drawMs.toFixed(0)} ms/frame</span>
		<button type="button" onclick={reset}>reset view</button>
	</div>
</div>

<style>
	.cloud {
		position: relative;
		width: 100%;
	}
	canvas {
		display: block;
		background: var(--bg-inset);
		border-radius: var(--radius);
		cursor: crosshair;
		touch-action: none;
	}
	.cloud:focus-visible canvas {
		outline: 2px solid var(--accent);
	}
	.tip {
		position: absolute;
		pointer-events: none;
		background: var(--bg-raised);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 3px 6px;
		font-size: 11px;
		max-width: 260px;
		white-space: normal;
	}
	.hud {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 11px;
		margin-top: 4px;
	}
	.hud button {
		font-size: 11px;
	}
</style>
