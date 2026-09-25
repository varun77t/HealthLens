import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";

/**
 * A rotating double helix, drawn as ink particles on a canvas.
 *
 * Decoration only: it says nothing about anyone's results and nothing reads it, so the
 * canvas is hidden from screen readers. The colour comes from the `--ink` token, so it
 * follows the theme, and the ends dissolve into loose particles instead of being cropped.
 *
 * It assembles once on arrival, then turns slowly. It stops drawing while off screen, has
 * its own pause button (anything that keeps moving must be stoppable), and starts paused
 * when the system asks for reduced motion.
 */

type Particle =
  | { k: 0; t: number; s: number; jr: number; ja: number; jz: number; sz: number; sx: number; sy: number; d: number }
  | { k: 1; t: number; u: number; jr: number; ja: number; sz: number; sx: number; sy: number; d: number }
  | {
      k: 2;
      t: number;
      s: number;
      dx: number;
      dy: number;
      life: number;
      rate: number;
      sz: number;
      sx: number;
      sy: number;
      d: number;
    };

// Seeded, so every visit draws the same molecule.
function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
function smooth(a: number, b: number, v: number) {
  v = clamp01((v - a) / (b - a));
  return v * v * (3 - 2 * v);
}

// Particles live in helix space: t along the axis (0 top, 1 bottom), an angle on the
// strand, and jitter. Screen positions are worked out every frame, so resizing is free.
function makeParticles(): Particle[] {
  const r = rng(20260925);
  const gauss = () => (r() + r() + r() - 1.5) / 1.5;
  const P: Particle[] = [];
  for (let i = 0; i < 2200; i++) {
    // two backbones
    P.push({
      k: 0, t: r(), s: i & 1 ? Math.PI : 0, jr: gauss() * 0.09, ja: gauss() * 0.012,
      jz: gauss() * 0.12, sz: 0.7 + r() * 0.9, sx: gauss(), sy: gauss(), d: r(),
    });
  }
  const RUNGS = 44;
  for (let i = 0; i < RUNGS; i++) {
    // base pairs, with a gap where the bases meet
    const t = (i + 0.5) / RUNGS;
    for (let j = 0; j < 20; j++) {
      const u = (j + r() * 0.6) / 20;
      if (Math.abs(u - 0.5) < 0.045) continue;
      P.push({
        k: 1, t, u, jr: gauss() * 0.02, ja: gauss() * 0.004, sz: 0.55 + r() * 0.6,
        sx: gauss(), sy: gauss(), d: r(),
      });
    }
  }
  for (let i = 0; i < 260; i++) {
    // loose particles drifting off the strands, mostly at the ends
    const end = r() < 0.75;
    P.push({
      k: 2, t: end ? (r() < 0.5 ? r() * 0.2 : 1 - r() * 0.2) : r(), s: r() * Math.PI * 2,
      dx: gauss(), dy: gauss(), life: r(), rate: 0.05 + r() * 0.08, sz: 0.5 + r() * 1.1,
      sx: gauss(), sy: gauss(), d: r(),
    });
  }
  return P;
}
const PARTS = makeParticles();

function readInk(): [number, number, number] {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim().split(/\s+/).map(Number);
  return v.length === 3 && v.every((n) => !isNaN(n)) ? [v[0], v[1], v[2]] : [0, 0, 0];
}

type Quad = { x: number; y: number; r: number; a: number };

export default function DnaHelix({ className = "" }: { className?: string }) {
  const boxRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const toggleRef = useRef<(() => void) | null>(null);
  const [playing, setPlaying] = useState(
    () => !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  useEffect(() => {
    const box = boxRef.current;
    const canvas = canvasRef.current;
    const g = canvas?.getContext("2d");
    if (!box || !canvas || !g) return;

    const root = document.documentElement;
    const darkMq = window.matchMedia("(prefers-color-scheme: dark)");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const ctx = gsap.context(() => {
      let W = 0, H = 0, ink = readInk(), visible = true, on = !reduce;
      const S = { phase: 0.6, speed: 0.42, run: on ? 1 : 0, build: 1, tilt: 0, time: 0 };
      const back: Quad[] = [], front: Quad[] = [];

      function size() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        W = box!.clientWidth;
        H = box!.clientHeight;
        canvas!.width = Math.round(W * dpr);
        canvas!.height = Math.round(H * dpr);
        g!.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      function paint(list: Quad[]) {
        for (const q of list) {
          g!.fillStyle = `rgba(${ink[0]},${ink[1]},${ink[2]},${q.a.toFixed(3)})`;
          if (q.r < 0.9) g!.fillRect(q.x - q.r, q.y - q.r, q.r * 2, q.r * 2);
          else {
            g!.beginPath();
            g!.arc(q.x, q.y, q.r, 0, 6.2832);
            g!.fill();
          }
        }
      }

      function draw() {
        if (!W || !H) return;
        g!.clearRect(0, 0, W, H);
        const alpha = 0.37 + S.tilt; // leans top-left to bottom-right
        const ax = Math.sin(alpha), ay = Math.cos(alpha), nx = ay, ny = -ax;
        const cx = W * 0.5, cy = H * 0.5;
        const len = Math.hypot(W, H) * 1.02;
        const R = Math.min(W * 0.2, H * 0.13, 96);
        const turns = len / (R * 4.1);
        const spread = Math.max(W, H) * 0.55;
        back.length = 0;
        front.length = 0;

        for (const p of PARTS) {
          const edge = smooth(0.62, 1, Math.abs(p.t - 0.5) * 2); // 0 in the middle, 1 at the ends
          const th = p.t * turns * Math.PI * 2 + S.phase;
          let off: number, z: number, a: number, along = p.t;

          if (p.k === 0) {
            const loose = 1 + edge * 7;
            off = Math.sin(th + p.s) + p.jr * loose;
            z = Math.cos(th + p.s) + p.jz * loose;
            along += p.ja * loose;
            a = 1 - edge * 0.9;
          } else if (p.k === 1) {
            const m = 1 - 2 * p.u;
            off = Math.sin(th) * m + p.jr * (1 + edge * 6);
            z = Math.cos(th) * m;
            along += p.ja;
            a = (0.7 - edge * 0.65) * (0.55 + 0.45 * Math.abs(Math.sin(th)));
          } else {
            const life = (p.life + S.time * p.rate) % 1;
            const reach = life * (0.8 + edge * 2.2);
            off = Math.sin(th + p.s) + p.dx * reach;
            z = Math.cos(th + p.s);
            along += p.dy * reach * 0.05;
            a = Math.sin(life * Math.PI) * (0.35 + edge * 0.45);
          }

          let x = cx + ax * (along - 0.5) * len + nx * off * R;
          let y = cy + ay * (along - 0.5) * len + ny * off * R;

          if (S.build < 1) {
            // assemble from a loose cloud, top to bottom
            const pr = clamp01(S.build * 1.6 - (p.t * 0.45 + p.d * 0.15));
            const e = 1 - Math.pow(1 - pr, 3);
            x += p.sx * spread * (1 - e);
            y += p.sy * spread * 0.6 * (1 - e);
            a *= pr;
          }
          if (a <= 0.01 || x < -4 || y < -4 || x > W + 4 || y > H + 4) continue;

          const depth = (z + 1) / 2; // 0 far, 1 near
          (z < 0 ? back : front).push({ x, y, r: p.sz * (0.55 + depth * 0.95), a: a * (0.22 + depth * 0.78) });
        }
        paint(back);
        paint(front);
      }

      function tick(_time: number, dt: number) {
        if (!visible) return;
        const s = Math.min(dt, 50) / 1000;
        S.phase += S.speed * S.run * s;
        S.time += s * S.run;
        draw();
      }
      gsap.ticker.add(tick);

      // Lean a little toward the pointer. Mouse only; it changes nothing you need to read.
      const tiltTo = gsap.quickTo(S, "tilt", { duration: 1.2, ease: "power3.out" });
      const host = box.parentElement ?? box;
      function onMove(e: PointerEvent) {
        if (!on || e.pointerType === "touch") return;
        const b = host.getBoundingClientRect();
        tiltTo(((e.clientX - b.left) / b.width - 0.5) * -0.14);
      }
      function onLeave() {
        tiltTo(0);
      }
      host.addEventListener("pointermove", onMove);
      host.addEventListener("pointerleave", onLeave);

      const io = new IntersectionObserver((en) => {
        visible = en[0].isIntersecting;
      });
      io.observe(box);
      const ro = new ResizeObserver(() => {
        size();
        draw();
      });
      ro.observe(box);
      function onTheme() {
        ink = readInk();
        draw();
      }
      const mo = new MutationObserver(onTheme);
      mo.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
      darkMq.addEventListener("change", onTheme);

      toggleRef.current = () => {
        on = !on;
        gsap.to(S, { run: on ? 1 : 0, duration: 0.6, ease: "power2.out", overwrite: "auto" });
        if (!on) tiltTo(0);
        setPlaying(on);
      };

      size();
      if (!reduce) {
        S.build = 0;
        S.speed = 3.2;
        gsap
          .timeline()
          .to(S, { build: 1, duration: 2.6, ease: "power2.out" }, 0.05)
          .to(S, { speed: 0.42, duration: 3.2, ease: "expo.out" }, 0.05);
      }
      draw();

      return () => {
        gsap.ticker.remove(tick);
        host.removeEventListener("pointermove", onMove);
        host.removeEventListener("pointerleave", onLeave);
        darkMq.removeEventListener("change", onTheme);
        io.disconnect();
        ro.disconnect();
        mo.disconnect();
        toggleRef.current = null;
      };
    }, box);

    return () => ctx.revert();
  }, []);

  const label = playing ? "Pause animation" : "Play animation";

  return (
    <div ref={boxRef} className={`relative ${className}`}>
      <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 block h-full w-full" />
      <button
        type="button"
        onClick={() => toggleRef.current?.()}
        aria-label={label}
        title={label}
        className="absolute bottom-2 right-2 inline-flex items-center justify-center rounded-md border border-line bg-canvas/80 p-2 text-muted transition-colors hover:bg-accent-soft hover:text-ink"
      >
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
            <path d="M8.5 5v14M15.5 5v14" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true">
            <path d="M7.5 4.8v14.4L19 12 7.5 4.8Z" />
          </svg>
        )}
      </button>
    </div>
  );
}
