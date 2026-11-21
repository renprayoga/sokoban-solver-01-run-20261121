// Push-optimal Sokoban solver: rooms are independent (a box pushed out never returns),
// so each room is solved by BFS over pushes with live-square and frozen-2x2 pruning.
// Work is budgeted by state counts only, never the clock, so runs are deterministic.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input', O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const RC = 350000, TC = 2500000;
let W = 0;
const X = [0, 0, -1, 1], Y = [-1, 1, 0, 0], S = ['u', 'd', 'l', 'r'], P = ['U', 'D', 'L', 'R'];

// A level in XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player).
function parse(t) {
  const r = t.split('\n').filter((x) => x.length), h = r.length, w = Math.max(...r.map((x) => x.length));
  const a = new Uint8Array(w * h).fill(1), g = [], b = [];
  let p = -1;
  r.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const c = row[x], i = y * w + x;
      if (c === '#') continue;
      a[i] = 0;
      if ('.*+'.includes(c)) g.push(i);
      if ('$*'.includes(c)) b.push(i);
      if ('@+'.includes(c)) p = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (!x || !y || x === w - 1 || y === h - 1) a[y * w + x] = 1;
  b.sort((m, n) => m - n);
  return { w, h, wall: a, goals: g, boxes: b, player: p };
}

// Squares from which a box alone can be pushed onto some goal (pulled back from the goals).
function live(L) {
  const { w, wall: a, goals: g } = L, v = new Uint8Array(a.length), q = g.slice();
  for (const x of g) v[x] = 1;
  for (let i = 0; i < q.length; i++) for (let k = 0; k < 4; k++) {
    const d = Y[k] * w + X[k], y = q[i] - d;
    if (y - d >= 0 && y - d < a.length && !a[y] && !a[y - d] && !v[y]) { v[y] = 1; q.push(y); }
  }
  return v;
}

// Flood fill from `f` around walls and boxes; returns the region's lowest square.
function reach(L, o, f, s, t, k) {
  const { w, wall: a } = L;
  let l = f, n = 0;
  k[n++] = f;
  s[f] = t;
  while (n) {
    const p = k[--n];
    if (p < l) l = p;
    for (let j = 0; j < 4; j++) {
      const q = p + Y[j] * w + X[j];
      if (!a[q] && !o[q] && s[q] !== t) { s[q] = t; k[n++] = q; }
    }
  }
  return l;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes.
function frozen(L, o, g, t) {
  const { w, wall: a } = L;
  for (const x of [t, t - 1, t - w, t - w - 1]) {
    const b = x + 1, c = x + w, d = c + 1;
    if ((a[x] || o[x]) && (a[b] || o[b]) && (a[c] || o[c]) && (a[d] || o[d]) &&
      ((o[x] && !g[x]) || (o[b] && !g[b]) || (o[c] && !g[c]) || (o[d] && !g[d]))) return 1;
  }
  return 0;
}

// The shortest walk from `f` to `t` around walls and boxes, in lower LURD letters.
function walk(L, o, f, t) {
  if (f === t) return '';
  const { w, wall: a } = L, v = new Int32Array(a.length).fill(-1), h = [], q = [f];
  v[f] = f;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === t) break;
    for (let k = 0; k < 4; k++) {
      const n = p + Y[k] * w + X[k];
      if (a[n] || o[n] || v[n] !== -1) continue;
      v[n] = p;
      h[n] = S[k];
      q.push(n);
    }
  }
  if (v[t] === -1) throw 0;
  let s = '';
  for (let p = t; p !== f; p = v[p]) s = h[p] + s;
  return s;
}

// Push-BFS over one room (or a whole level as fallback): the first solved state generated
// is push-optimal. Returns the moves, or null when the state cap or work budget runs out.
function search(L, cap) {
  const { w, wall: a, goals: g } = L, n = a.length, go = new Uint8Array(n), lv = live(L);
  for (const x of g) go[x] = 1;
  if (L.boxes.some((b) => !lv[b])) return null;
  const ok = (b) => b.every((x) => go[x]);
  if (ok(L.boxes)) return '';
  const o = new Uint8Array(n), s = new Int32Array(n), m = new Int32Array(n), k = new Int32Array(n);
  let t = 0;
  const B = [L.boxes.slice()], Q = [L.player], F = [-1], A = [-1], T = [-1], idx = new Map();
  for (const b of L.boxes) o[b] = 1;
  let key = reach(L, o, L.player, s, ++t, k);
  for (const b of L.boxes) { o[b] = 0; key = key * n + b; }
  idx.set(key, 0);
  let f = -1;
  for (let hd = 0; f < 0 && hd < B.length; hd++) {
    const bx = B[hd];
    for (const b of bx) o[b] = 1;
    reach(L, o, Q[hd], s, ++t, k);
    const hr = t;
    for (let bi = 0; bi < bx.length && f < 0; bi++) {
      const b = bx[bi];
      for (let j = 0; j < 4; j++) {
        const d = Y[j] * w + X[j], u = b + d;
        if (s[b - d] !== hr || a[u] || o[u] || !lv[u]) continue;
        o[b] = 0;
        o[u] = 1;
        const fr = frozen(L, o, go, u);
        let ky = -1, nx = null;
        if (!fr) {
          nx = bx.slice();
          nx[bi] = u;
          nx.sort((x, y) => x - y);
          ky = reach(L, o, b, m, ++t, k);
          for (const z of nx) ky = ky * n + z;
        }
        o[u] = 0;
        o[b] = 1;
        if (fr || idx.has(ky)) continue;
        if (B.length >= cap || W >= TC) {
          for (const x of bx) o[x] = 0;
          return null;
        }
        idx.set(ky, B.length);
        B.push(nx);
        Q.push(b);
        F.push(hd);
        A.push(b);
        T.push(u);
        W++;
        if (ok(nx)) { f = B.length - 1; break; }
      }
    }
    for (const x of bx) o[x] = 0;
  }
  if (f < 0) return null;
  const path = [];
  for (let x = f; x > 0; x = F[x]) path.push(x);
  path.reverse();
  let v = '', pl = L.player, bx = L.boxes.slice();
  for (const x of path) {
    const fr = A[x], to = T[x], d = to - fr, j = X.findIndex((dx, i) => Y[i] * w + dx === d);
    for (const b of bx) o[b] = 1;
    v += walk(L, o, pl, fr - d);
    for (const b of bx) o[b] = 0;
    v += P[j];
    pl = fr;
    bx = B[x];
  }
  return v;
}

// The level's rooms: cut the hallway row's dead squares (the player's row, where a box
// reaches no goal) out of the floor; each remaining part holding boxes is a room. Null
// when a box starts in the hallway or a part's boxes and goals do not match.
function rooms(L) {
  const { w, wall: a, goals: g, boxes: b, player: p } = L, lv = live(L);
  const row = Math.floor(p / w), c = new Uint8Array(a.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!a[i] && !lv[i]) c[i] = 1;
  if (b.some((x) => c[x])) return null;
  const part = new Int32Array(a.length).fill(-1), R = [];
  for (let s = 0; s < a.length; s++) {
    if (a[s] || c[s] || part[s] >= 0) continue;
    const q = [s];
    part[s] = R.length;
    for (let i = 0; i < q.length; i++) for (let k = 0; k < 4; k++) {
      const n = q[i] + Y[k] * w + X[k];
      if (!a[n] && !c[n] && part[n] < 0) { part[n] = R.length; q.push(n); }
    }
    R.push({ boxes: [], goals: [] });
  }
  for (const x of b) R[part[x]].boxes.push(x);
  for (const x of g) if (part[x] >= 0) R[part[x]].goals.push(x);
  if (g.some((x) => part[x] < 0) || R.some((r) => r.boxes.length !== r.goals.length)) return null;
  return R.filter((r) => r.boxes.length);
}

// Solve one level: room by room, the player walking on from where the last room left it.
function solve(t) {
  const L = parse(t);
  if (!L.boxes.length) return '';
  const R = rooms(L);
  if (!R) return '';
  const done = new Set();
  let pl = L.player, v = '';
  for (let i = 0; i < R.length; i++) {
    const r = R[i], a = L.wall.slice();
    for (let j = 0; j < R.length; j++) {
      if (j === i) continue;
      for (const b of done.has(j) ? R[j].goals : R[j].boxes) a[b] = 1;
    }
    const s = search(
      { w: L.w, h: L.h, wall: a, goals: r.goals.slice().sort((x, y) => x - y), boxes: r.boxes.slice().sort((x, y) => x - y), player: pl },
      RC,
    );
    if (s === null) return '';
    v += s;
    for (const ch of s) {
      const c = ch.toLowerCase();
      pl += c === 'u' ? -L.w : c === 'd' ? L.w : c === 'l' ? -1 : 1;
    }
    done.add(i);
  }
  return v;
}

for (const id of readdirSync(I).sort()) {
  let v = '';
  try {
    v = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    v = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), v + '\n');
}
