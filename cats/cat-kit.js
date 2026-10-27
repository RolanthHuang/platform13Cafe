/*
 * Platform 13 resident cats — photo-matched rebuild (detail v14, 2026-10-26).
 *
 * createCatKit(THREE, TSL, U) → { build(opts) }
 *   THREE : namespace with core classes (Mesh, Group, Vector3, ... MeshPhysicalNodeMaterial ...)
 *   TSL   : three/tsl namespace
 *   U     : { mergeGeometries, furTexture }  (furTexture = dark fur photo used for hair detail)
 *
 * Every cat is a single continuous signed-distance body (skull, cheeks, muzzle, neck, torso,
 * legs, paws, tail) polygonised with block-sparse surface nets, painted per vertex from the
 * reference photos, plus separate shell ears, glossy eyes, nose leather, mouth line,
 * whiskers and (for the two black cats) gathered petal "scrunchie" collars.
 *
 * Local frame used while modelling: +z = nose direction, +y = up, +x = the cat's left,
 * y = 0 is the surface the cat lies on. build() wraps it so the returned group faces -z
 * like the previous cats did (scene placement and camera views stay valid).
 */
export function createCatKit(THREE, TSL, U = {}) {
  const { Vector3, Quaternion, Euler, Matrix4, Color, BufferGeometry, Float32BufferAttribute, Mesh, Group } = THREE;

  /* ------------------------------------------------------------------ math */
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const sstep = (a, b, x) => { let t = (x - a) / (b - a); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); };
  const smin = (a, b, k) => { const h = k - Math.abs(a - b); if (h <= 0) return a < b ? a : b; const q = h / k; return (a < b ? a : b) - q * q * k * 0.25; };
  const smax = (a, b, k) => -smin(-a, -b, k);
  const V = (a) => new Vector3(a[0], a[1], a[2]);
  const C = (hex) => new Color(hex).toArray();
  const mixc = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const scalec = (a, s) => [a[0] * s, a[1] * s, a[2] * s];

  function hash(i, j, k, s) {
    let h = Math.imul(i | 0, 374761393) ^ Math.imul(j | 0, 668265263) ^ Math.imul(k | 0, 1274126177) ^ Math.imul((s | 0) + 1, 2246822519);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
  }
  function vnoise(x, y, z, s = 0) {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    let u = x - xi, v = y - yi, w = z - zi;
    u = u * u * (3 - 2 * u); v = v * v * (3 - 2 * v); w = w * w * (3 - 2 * w);
    const a = lerp(hash(xi, yi, zi, s), hash(xi + 1, yi, zi, s), u), b = lerp(hash(xi, yi + 1, zi, s), hash(xi + 1, yi + 1, zi, s), u);
    const c = lerp(hash(xi, yi, zi + 1, s), hash(xi + 1, yi, zi + 1, s), u), d = lerp(hash(xi, yi + 1, zi + 1, s), hash(xi + 1, yi + 1, zi + 1, s), u);
    return lerp(lerp(a, b, v), lerp(c, d, v), w);
  }
  const fbm = (x, y, z, s = 0) => vnoise(x, y, z, s) * 0.55 + vnoise(x * 2.03 + 3.1, y * 2.03, z * 2.03, s + 7) * 0.3 + vnoise(x * 4.1, y * 4.1 + 1.7, z * 4.1, s + 13) * 0.15;
  function rng(seed) { let s = seed >>> 0; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }
  // distance from point (px,py) to polyline pts [[x,y],...]
  function dPoly(px, py, pts) {
    let best = 1e9;
    for (let i = 0; i < pts.length - 1; i++) {
      const ax = pts[i][0], ay = pts[i][1], bx = pts[i + 1][0], by = pts[i + 1][1];
      const ex = bx - ax, ey = by - ay, l = ex * ex + ey * ey;
      const t = l > 0 ? clamp(((px - ax) * ex + (py - ay) * ey) / l, 0, 1) : 0;
      const dx = px - ax - ex * t, dy = py - ay - ey * t, d = dx * dx + dy * dy;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }

  /* ------------------------------------------------------------ primitives */
  const K_ELL = 1, K_CONE = 2, K_SWEEP = 3, K_EYE = 4;
  const CAP = 0.02;
  function axesOf(q) {
    const e = new Matrix4().makeRotationFromQuaternion(q).elements;
    return [e[0], e[1], e[2], e[4], e[5], e[6], e[8], e[9], e[10]]; // rows = local axes in world
  }
  function ell(c, r, q, k, tag, extra = {}) {
    const m = q ? axesOf(q) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const ex = Math.hypot(r[0] * m[0], r[1] * m[3], r[2] * m[6]);
    const ey = Math.hypot(r[0] * m[1], r[1] * m[4], r[2] * m[7]);
    const ez = Math.hypot(r[0] * m[2], r[1] * m[5], r[2] * m[8]);
    return Object.assign({ kind: K_ELL, k, tag, cx: c[0], cy: c[1], cz: c[2], m, rx: r[0], ry: r[1], rz: r[2],
      irx: 1 / r[0], iry: 1 / r[1], irz: 1 / r[2], irx2: 1 / (r[0] * r[0]), iry2: 1 / (r[1] * r[1]), irz2: 1 / (r[2] * r[2]),
      rmin: Math.min(r[0], r[1], r[2]), box: [c[0] - ex, c[1] - ey, c[2] - ez, c[0] + ex, c[1] + ey, c[2] + ez] }, extra);
  }
  function cone(a, b, ra, rb, k, tag, extra = {}) {
    const bax = b[0] - a[0], bay = b[1] - a[1], baz = b[2] - a[2];
    const l2 = bax * bax + bay * bay + baz * baz, rr = ra - rb;
    return Object.assign({ kind: K_CONE, k, tag, ax: a[0], ay: a[1], az: a[2], bx: b[0], by: b[1], bz: b[2], bax, bay, baz, l2, rr, a2: l2 - rr * rr, il2: 1 / l2, ra, rb,
      len: Math.sqrt(l2), box: [Math.min(a[0] - ra, b[0] - rb), Math.min(a[1] - ra, b[1] - rb), Math.min(a[2] - ra, b[2] - rb),
        Math.max(a[0] + ra, b[0] + rb), Math.max(a[1] + ra, b[1] + rb), Math.max(a[2] + ra, b[2] + rb)] }, extra);
  }
  // generalised cylinder: smooth spine with egg-shaped sections (half width w, dorsal hu, ventral hd)
  function sweep(ctrl, n, k, tag, extra = {}) {
    const curve = new THREE.CatmullRomCurve3(ctrl.map((c) => V(c.p)), false, 'centripetal');
    const m = ctrl.length;
    const sc = (key, t) => { // catmull-rom on scalar keyed by control index
      const f = t * (m - 1), i = Math.min(m - 2, Math.floor(f)), u = f - i;
      const p0 = ctrl[Math.max(0, i - 1)][key], p1 = ctrl[i][key], p2 = ctrl[i + 1][key], p3 = ctrl[Math.min(m - 1, i + 2)][key];
      const u2 = u * u, u3 = u2 * u;
      return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3);
    };
    const upAt = (t) => {
      const f = t * (m - 1), i = Math.min(m - 2, Math.floor(f)), u = sstep(0, 1, f - i);
      const a = ctrl[i].up || [0, 1, 0], b = ctrl[i + 1].up || [0, 1, 0];
      return new Vector3(lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerp(a[2], b[2], u));
    };
    const P = new Float64Array(n * 3), T = new Float64Array(n * 3), Up = new Float64Array(n * 3), Sd = new Float64Array(n * 3);
    const W = new Float64Array(n), HU = new Float64Array(n), HD = new Float64Array(n), L = new Float64Array(n);
    const box = [1e9, 1e9, 1e9, -1e9, -1e9, -1e9];
    let acc = 0;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1), p = curve.getPoint(t), tg = curve.getTangent(t).normalize();
      const up = upAt(t); up.addScaledVector(tg, -up.dot(tg)).normalize();
      const sd = new Vector3().crossVectors(tg, up).normalize();
      P[i * 3] = p.x; P[i * 3 + 1] = p.y; P[i * 3 + 2] = p.z;
      T[i * 3] = tg.x; T[i * 3 + 1] = tg.y; T[i * 3 + 2] = tg.z;
      Up[i * 3] = up.x; Up[i * 3 + 1] = up.y; Up[i * 3 + 2] = up.z;
      Sd[i * 3] = sd.x; Sd[i * 3 + 1] = sd.y; Sd[i * 3 + 2] = sd.z;
      W[i] = sc('w', t); HU[i] = sc('hu', t); HD[i] = sc('hd', t);
      if (i) acc += Math.hypot(p.x - P[i * 3 - 3], p.y - P[i * 3 - 2], p.z - P[i * 3 - 1]);
      L[i] = acc;
      const r = Math.max(W[i], HU[i], HD[i]);
      box[0] = Math.min(box[0], p.x - r); box[1] = Math.min(box[1], p.y - r); box[2] = Math.min(box[2], p.z - r);
      box[3] = Math.max(box[3], p.x + r); box[4] = Math.max(box[4], p.y + r); box[5] = Math.max(box[5], p.z + r);
    }
    return Object.assign({ kind: K_SWEEP, k, tag, n, stride: Math.max(2, Math.round(Math.sqrt(n))), P, T, Up, Sd, W, HU, HD, L, total: acc, box, capStart: true, capEnd: true }, extra);
  }

  // almond / lemon shaped socket carved around each eye (extruded along the eye axis)
  function eyeCarve(c, q, w, h, depth, k, pinch) {
    const p = ell(c, [w, h * 1.05, depth], q, k, 'eye');
    p.kind = K_EYE; p.w = w; p.h = h; p.depth = depth; p.pinch = pinch;
    return p;
  }
  // local coordinate scratch filled by evalPrim when wanted
  const LOC = { t: 0, s: 0, phi: 0, u: 0, v: 0, a: 0, lx: 0, ly: 0, lz: 0, tx: 0, ty: 0, tz: 1, ux: 0, uy: 1, uz: 0 };
  function evalPrim(pr, x, y, z, wantLoc) {
    if (pr.kind === K_EYE) {
      const dx = x - pr.cx, dy = y - pr.cy, dz = z - pr.cz, m = pr.m;
      const lx = m[0] * dx + m[1] * dy + m[2] * dz, ly = m[3] * dx + m[4] * dy + m[5] * dz, lz = m[6] * dx + m[7] * dy + m[8] * dz;
      const xn = lx / pr.w;
      const hh = pr.h * Math.max(0.08, 1 - pr.pinch * xn * xn) * (ly > 0 ? 0.97 : 1.05);
      const ax = xn, ay = ly / hh, bx = lx / (pr.w * pr.w), by = ly / (hh * hh);
      const k0 = Math.sqrt(ax * ax + ay * ay), k1 = Math.sqrt(bx * bx + by * by);
      const d2 = k1 > 1e-12 ? (k0 * (k0 - 1)) / k1 : -hh;
      const dz2 = Math.abs(lz) - pr.depth;
      return d2 > dz2 ? d2 : dz2;
    }
    if (pr.kind === K_ELL) {
      const dx = x - pr.cx, dy = y - pr.cy, dz = z - pr.cz, m = pr.m;
      const lx = m[0] * dx + m[1] * dy + m[2] * dz, ly = m[3] * dx + m[4] * dy + m[5] * dz, lz = m[6] * dx + m[7] * dy + m[8] * dz;
      if (wantLoc) { LOC.lx = lx * pr.irx; LOC.ly = ly * pr.iry; LOC.lz = lz * pr.irz; }
      const ax = lx * pr.irx, ay = ly * pr.iry, az = lz * pr.irz;
      const bx = lx * pr.irx2, by = ly * pr.iry2, bz = lz * pr.irz2;
      const k0 = Math.sqrt(ax * ax + ay * ay + az * az), k1 = Math.sqrt(bx * bx + by * by + bz * bz);
      return k1 > 1e-12 ? (k0 * (k0 - 1)) / k1 : -pr.rmin;
    }
    if (pr.kind === K_CONE) {
      const pax = x - pr.ax, pay = y - pr.ay, paz = z - pr.az;
      const yv = pax * pr.bax + pay * pr.bay + paz * pr.baz, zv = yv - pr.l2;
      if (wantLoc) LOC.t = clamp(yv * pr.il2, 0, 1);
      const qx = pax * pr.l2 - pr.bax * yv, qy = pay * pr.l2 - pr.bay * yv, qz = paz * pr.l2 - pr.baz * yv;
      const x2 = qx * qx + qy * qy + qz * qz, y2 = yv * yv * pr.l2, z2 = zv * zv * pr.l2;
      const kk = Math.sign(pr.rr) * pr.rr * pr.rr * x2;
      if (Math.sign(zv) * pr.a2 * z2 > kk) return Math.sqrt(x2 + z2) * pr.il2 - pr.rb;
      if (Math.sign(yv) * pr.a2 * y2 < kk) return Math.sqrt(x2 + y2) * pr.il2 - pr.ra;
      return (Math.sqrt(x2 * pr.a2 * pr.il2) + yv * pr.rr) * pr.il2 - pr.ra;
    }
    // sweep: coarse-to-fine closest sample, then the two neighbouring segments
    const n = pr.n, P = pr.P, ST = pr.stride;
    let bi = 0, bd = 1e18;
    for (let i = 0; i < n; i += ST) {
      const dx = x - P[i * 3], dy = y - P[i * 3 + 1], dz = z - P[i * 3 + 2], d = dx * dx + dy * dy + dz * dz;
      if (d < bd) { bd = d; bi = i; }
    }
    {
      const lo = bi > ST ? bi - ST : 0, hi = bi + ST < n ? bi + ST : n - 1;
      for (let i = lo; i <= hi; i++) {
        const dx = x - P[i * 3], dy = y - P[i * 3 + 1], dz = z - P[i * 3 + 2], d = dx * dx + dy * dy + dz * dz;
        if (d < bd) { bd = d; bi = i; }
      }
    }
    let bestT = bi, bestD = 1e18;
    for (let s = bi > 0 ? bi - 1 : 0, se = bi < n - 2 ? bi : n - 2; s <= se; s++) {
      const ax = P[s * 3], ay = P[s * 3 + 1], az = P[s * 3 + 2];
      const ex = P[s * 3 + 3] - ax, ey = P[s * 3 + 4] - ay, ez = P[s * 3 + 5] - az;
      let u = ((x - ax) * ex + (y - ay) * ey + (z - az) * ez) / (ex * ex + ey * ey + ez * ez);
      u = u < 0 ? 0 : u > 1 ? 1 : u;
      const dx = x - ax - ex * u, dy = y - ay - ey * u, dz = z - az - ez * u, d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) { bestD = d; bestT = s + u; }
    }
    const i0 = bestT >= n - 1 ? n - 2 : Math.floor(bestT), f = bestT - i0, g = 1 - f, a0 = i0 * 3, a1 = a0 + 3;
    // cubic (catmull-rom) weights → C1 centre line and sections (no ribbing at the samples)
    const ff = f * f, fff = ff * f;
    const w0 = 0.5 * (-f + 2 * ff - fff), w1 = 0.5 * (2 - 5 * ff + 3 * fff), w2 = 0.5 * (f + 4 * ff - 3 * fff), w3 = 0.5 * (fff - ff);
    const im = i0 > 0 ? i0 - 1 : 0, ip = i0 + 2 < n ? i0 + 2 : n - 1, am = im * 3, ap = ip * 3;
    const TT = pr.T, UP = pr.Up, SD = pr.Sd, WW = pr.W, HU = pr.HU, HD = pr.HD;
    const qx = x - (P[am] * w0 + P[a0] * w1 + P[a1] * w2 + P[ap] * w3);
    const qy = y - (P[am + 1] * w0 + P[a0 + 1] * w1 + P[a1 + 1] * w2 + P[ap + 1] * w3);
    const qz = z - (P[am + 2] * w0 + P[a0 + 2] * w1 + P[a1 + 2] * w2 + P[ap + 2] * w3);
    const along = qx * (TT[a0] * g + TT[a1] * f) + qy * (TT[a0 + 1] * g + TT[a1 + 1] * f) + qz * (TT[a0 + 2] * g + TT[a1 + 2] * f);
    const u = qx * (SD[a0] * g + SD[a1] * f) + qy * (SD[a0 + 1] * g + SD[a1 + 1] * f) + qz * (SD[a0 + 2] * g + SD[a1 + 2] * f);
    const v = qx * (UP[a0] * g + UP[a1] * f) + qy * (UP[a0 + 1] * g + UP[a1 + 1] * f) + qz * (UP[a0 + 2] * g + UP[a1 + 2] * f);
    const w = WW[im] * w0 + WW[i0] * w1 + WW[i0 + 1] * w2 + WW[ip] * w3;
    const b = v > 0 ? HU[im] * w0 + HU[i0] * w1 + HU[i0 + 1] * w2 + HU[ip] * w3 : HD[im] * w0 + HD[i0] * w1 + HD[i0 + 1] * w2 + HD[ip] * w3;
    if (wantLoc) {
      LOC.t = bestT / (n - 1); LOC.s = pr.L[i0] * g + pr.L[i0 + 1] * f; LOC.phi = Math.atan2(u, v);
      LOC.tx = TT[a0] * g + TT[a1] * f; LOC.ty = TT[a0 + 1] * g + TT[a1 + 1] * f; LOC.tz = TT[a0 + 2] * g + TT[a1 + 2] * f;
      LOC.ux = UP[a0] * g + UP[a1] * f; LOC.uy = UP[a0 + 1] * g + UP[a1 + 1] * f; LOC.uz = UP[a0 + 2] * g + UP[a1 + 2] * f;
      LOC.u = u / w; LOC.v = v / b; LOC.a = along;
    }
    let e = 0;
    if (pr.capStart && bestT <= 0 && along < 0) e = -along;
    else if (pr.capEnd && bestT >= n - 1 && along > 0) e = along;
    if (e > 0) { // rounded ellipsoidal cap
      const cz = Math.min(w, b) * (pr.capScale || 1);
      const ax = u / w, ay = v / b, az = e / cz, bx = u / (w * w), by = v / (b * b), bz = e / (cz * cz);
      const k0 = Math.sqrt(ax * ax + ay * ay + az * az), k1 = Math.sqrt(bx * bx + by * by + bz * bz);
      return k1 > 1e-12 ? (k0 * (k0 - 1)) / k1 : -Math.min(w, b);
    }
    const ax = u / w, ay = v / b, bx = u / (w * w), by = v / (b * b);
    const k0 = Math.sqrt(ax * ax + ay * ay), k1 = Math.sqrt(bx * bx + by * by);
    return k1 > 1e-12 ? (k0 * (k0 - 1)) / k1 : -Math.min(w, b);
  }

  function makeSDF(prims, subs, ground) {
    for (const p of prims.concat(subs)) {
      const m = CAP + p.k;
      p.bb = [p.box[0] - m, p.box[1] - m, p.box[2] - m, p.box[3] + m, p.box[4] + m, p.box[5] + m];
    }
    const evalList = (L, nL, S, nS) => (x, y, z) => {
      let d = 1e9;
      for (let i = 0; i < nL; i++) {
        const p = L[i], bb = p.bb;
        if (x < bb[0] || y < bb[1] || z < bb[2] || x > bb[3] || y > bb[4] || z > bb[5]) continue;
        const di = evalPrim(p, x, y, z, false);
        d = d === 1e9 ? di : smin(d, di, p.k);
      }
      if (d > CAP) d = CAP;
      for (let i = 0; i < nS; i++) {
        const p = S[i], bb = p.box, k = p.k;
        if (x < bb[0] - k || y < bb[1] - k || z < bb[2] - k || x > bb[3] + k || y > bb[4] + k || z > bb[5] + k) continue;
        d = smax(d, -evalPrim(p, x, y, z, false), k);
      }
      if (ground) d = smax(d, ground.y - y, ground.k);
      return d;
    };
    const f = evalList(prims, prims.length, subs, subs.length);
    // evaluator specialised to an axis-aligned box: drops primitives that cannot reach the surface there
    f.local = (lo, hi) => {
      const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2;
      const hd = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 1.25;
      let U = CAP;
      const cand = [];
      for (const p of prims) {
        const b = p.box;
        const gx = Math.max(0, b[0] - hi[0], lo[0] - b[3]), gy = Math.max(0, b[1] - hi[1], lo[1] - b[4]), gz = Math.max(0, b[2] - hi[2], lo[2] - b[5]);
        const lb = Math.hypot(gx, gy, gz);
        if (lb > CAP + p.k) continue;
        const dc = evalPrim(p, cx, cy, cz, false);
        if (dc + hd < U) U = dc + hd;
        cand.push(p, Math.max(lb, dc - hd));
      }
      const L = [];
      for (let i = 0; i < cand.length; i += 2) if (cand[i + 1] <= U + cand[i].k + 1e-4) L.push(cand[i]);
      const S = subs.filter((p) => { const b = p.box, k = p.k; return !(b[0] - k > hi[0] || b[1] - k > hi[1] || b[2] - k > hi[2] || b[3] + k < lo[0] || b[4] + k < lo[1] || b[5] + k < lo[2]); });
      const lf = evalList(L, L.length, S, S.length);
      lf.count = L.length;
      return lf;
    };
    f.prims = prims; f.subs = subs;
    return f;
  }
  function gradient(f, x, y, z, e) {
    // tetrahedral gradient
    const a = f(x + e, y - e, z - e), b = f(x - e, y - e, z + e), c = f(x - e, y + e, z - e), d = f(x + e, y + e, z + e);
    const gx = a - b - c + d, gy = -a - b + c + d, gz = -a + b - c + d;
    const l = Math.hypot(gx, gy, gz) || 1;
    return [gx / l, gy / l, gz / l, l / (4 * e)];
  }
  // march from p along dir until crossing the surface (returns point or null)
  function rayHit(f, p, dir, maxLen = 0.2, step = 0.0006) {
    let x = p[0], y = p[1], z = p[2], prev = f(x, y, z);
    for (let t = 0; t < maxLen; t += step) {
      const nx = x + dir[0] * step, ny = y + dir[1] * step, nz = z + dir[2] * step, d = f(nx, ny, nz);
      if ((prev > 0) !== (d > 0)) {
        const s = prev / (prev - d);
        return [x + (nx - x) * s, y + (ny - y) * s, z + (nz - z) * s];
      }
      x = nx; y = ny; z = nz; prev = d;
    }
    return null;
  }

  /* --------------------------------------------- block-sparse surface nets */
  function polygonize(f, bmin, bmax, h) {
    const SH = 2, B = 1 << SH, BM = B - 1, BS = B + 1, BS2 = BS * BS;
    const nbx = Math.ceil((bmax[0] - bmin[0]) / (h * B)), nby = Math.ceil((bmax[1] - bmin[1]) / (h * B)), nbz = Math.ceil((bmax[2] - bmin[2]) / (h * B));
    const NX = nbx * B, NY = nby * B, NZ = nbz * B;
    const nb = nbx * nby * nbz;
    const vals = new Array(nb).fill(null), locals = new Array(nb).fill(null);
    const R = (h * B * Math.sqrt(3)) / 2, thr = R * 1.35 + h;
    // coarse corner values (block corners)
    const cnx = nbx + 1, cny = nby + 1, cnz = nbz + 1, corner = new Float32Array(cnx * cny * cnz);
    for (let k = 0; k < cnz; k++) for (let j = 0; j < cny; j++) for (let i = 0; i < cnx; i++)
      corner[(k * cny + j) * cnx + i] = f(bmin[0] + i * B * h, bmin[1] + j * B * h, bmin[2] + k * B * h);
    for (let bk = 0; bk < nbz; bk++) for (let bj = 0; bj < nby; bj++) for (let bi = 0; bi < nbx; bi++) {
      // quick reject using corners: all same sign and far
      let mn = 1e9, s0 = 0, s1 = 0;
      for (let q = 0; q < 8; q++) {
        const v = corner[((bk + (q >> 2)) * cny + bj + ((q >> 1) & 1)) * cnx + bi + (q & 1)];
        const av = v < 0 ? -v : v;
        if (av < mn) mn = av;
        if (v < 0) s1++; else s0++;
      }
      if (s0 && s1) { /* surface crosses */ }
      else if (mn > 2 * R + h) continue;
      else {
        const dc = f(bmin[0] + (bi + 0.5) * B * h, bmin[1] + (bj + 0.5) * B * h, bmin[2] + (bk + 0.5) * B * h);
        if ((dc < 0 ? -dc : dc) > thr) continue;
      }
      const bIdx = (bk * nby + bj) * nbx + bi;
      const lo = [bmin[0] + bi * B * h - h, bmin[1] + bj * B * h - h, bmin[2] + bk * B * h - h];
      const lf = f.local ? f.local(lo, [lo[0] + (B + 2) * h, lo[1] + (B + 2) * h, lo[2] + (B + 2) * h]) : f;
      locals[bIdx] = lf;
      const L = bi > 0 ? vals[bIdx - 1] : null, D = bj > 0 ? vals[bIdx - nbx] : null, K = bk > 0 ? vals[bIdx - nbx * nby] : null;
      const a = new Float32Array(BS * BS2);
      let o = 0;
      for (let k = 0; k < BS; k++) {
        const z = bmin[2] + (bk * B + k) * h;
        for (let j = 0; j < BS; j++) {
          const y = bmin[1] + (bj * B + j) * h;
          for (let i = 0; i < BS; i++, o++) {
            if (i === 0 && L) a[o] = L[k * BS2 + j * BS + B];
            else if (j === 0 && D) a[o] = D[k * BS2 + B * BS + i];
            else if (k === 0 && K) a[o] = K[B * BS2 + j * BS + i];
            else a[o] = lf(bmin[0] + (bi * B + i) * h, y, z);
          }
        }
      }
      vals[bIdx] = a;
    }
    const verts = new Array(nb).fill(null);
    const pos = [], vblock = [];
    const EA = [0, 2, 4, 6, 0, 1, 4, 5, 0, 1, 2, 3], EB = [1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7];
    const cv = new Float32Array(8);
    for (let b = 0; b < nb; b++) {
      const a = vals[b];
      if (!a) continue;
      const bi = b % nbx, bj = Math.floor(b / nbx) % nby, bk = Math.floor(b / (nbx * nby));
      const vb = new Int32Array(B * B * B).fill(-1);
      for (let k = 0; k < B; k++) for (let j = 0; j < B; j++) for (let i = 0; i < B; i++) {
        let mask = 0;
        for (let q = 0; q < 8; q++) {
          const v = a[(k + (q >> 2)) * BS2 + (j + ((q >> 1) & 1)) * BS + i + (q & 1)];
          cv[q] = v; if (v < 0) mask |= 1 << q;
        }
        if (mask === 0 || mask === 255) continue;
        let sx = 0, sy = 0, sz = 0, cnt = 0;
        for (let e = 0; e < 12; e++) {
          const p = EA[e], q = EB[e], va = cv[p], vq = cv[q];
          if ((va < 0) === (vq < 0)) continue;
          const t = va / (va - vq);
          sx += (p & 1) + ((q & 1) - (p & 1)) * t; sy += ((p >> 1) & 1) + (((q >> 1) & 1) - ((p >> 1) & 1)) * t; sz += (p >> 2) + ((q >> 2) - (p >> 2)) * t; cnt++;
        }
        vb[(k * B + j) * B + i] = pos.length / 3;
        vblock.push(b);
        pos.push(bmin[0] + (bi * B + i + sx / cnt) * h, bmin[1] + (bj * B + j + sy / cnt) * h, bmin[2] + (bk * B + k + sz / cnt) * h);
      }
      verts[b] = vb;
    }
    const look = (gi, gj, gk) => {
      if (gi < 0 || gj < 0 || gk < 0 || gi >= NX || gj >= NY || gk >= NZ) return -1;
      const vb = verts[((gk >> SH) * nby + (gj >> SH)) * nbx + (gi >> SH)];
      return vb ? vb[(((gk & BM) * B) + (gj & BM)) * B + (gi & BM)] : -1;
    };
    const idx = [];
    const quad = (a, b, c, d) => {
      if (a < 0 || b < 0 || c < 0 || d < 0) return;
      const d1 = (pos[a * 3] - pos[c * 3]) ** 2 + (pos[a * 3 + 1] - pos[c * 3 + 1]) ** 2 + (pos[a * 3 + 2] - pos[c * 3 + 2]) ** 2;
      const d2 = (pos[b * 3] - pos[d * 3]) ** 2 + (pos[b * 3 + 1] - pos[d * 3 + 1]) ** 2 + (pos[b * 3 + 2] - pos[d * 3 + 2]) ** 2;
      if (d1 < d2) idx.push(a, b, c, a, c, d); else idx.push(a, b, d, b, c, d);
    };
    for (let b = 0; b < nb; b++) {
      const a = vals[b];
      if (!a) continue;
      const bi = b % nbx, bj = Math.floor(b / nbx) % nby, bk = Math.floor(b / (nbx * nby));
      for (let k = 0; k < B; k++) for (let j = 0; j < B; j++) for (let i = 0; i < B; i++) {
        const v0 = a[k * BS2 + j * BS + i], in0 = v0 < 0;
        const gi = bi * B + i, gj = bj * B + j, gk = bk * B + k;
        if ((a[k * BS2 + j * BS + i + 1] < 0) !== in0) {
          const A = look(gi, gj - 1, gk - 1), Bq = look(gi, gj, gk - 1), Cq = look(gi, gj, gk), D = look(gi, gj - 1, gk);
          if (in0) quad(A, Bq, Cq, D); else quad(D, Cq, Bq, A);
        }
        if ((a[k * BS2 + (j + 1) * BS + i] < 0) !== in0) {
          const A = look(gi - 1, gj, gk - 1), Bq = look(gi - 1, gj, gk), Cq = look(gi, gj, gk), D = look(gi, gj, gk - 1);
          if (in0) quad(A, Bq, Cq, D); else quad(D, Cq, Bq, A);
        }
        if ((a[(k + 1) * BS2 + j * BS + i] < 0) !== in0) {
          const A = look(gi - 1, gj - 1, gk), Bq = look(gi, gj - 1, gk), Cq = look(gi, gj, gk), D = look(gi - 1, gj, gk);
          if (in0) quad(A, Bq, Cq, D); else quad(D, Cq, Bq, A);
        }
      }
    }
    return { pos: new Float32Array(pos), idx: new Uint32Array(idx), vblock, locals };
  }

  // project verts onto the surface (one Newton step) + gradient normals
  function refine(f, pos, h, mesh) {
    const n = pos.length / 3, nor = new Float32Array(n * 3), e = h * 0.35;
    for (let i = 0; i < n; i++) {
      let x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const lf = mesh && mesh.locals[mesh.vblock[i]] || f;
      const d = lf(x, y, z);
      const g = gradient(lf, x, y, z, e);
      const step = clamp(d / Math.max(g[3], 0.3), -h * 0.75, h * 0.75);
      x -= g[0] * step; y -= g[1] * step; z -= g[2] * step;
      pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
      nor[i * 3] = g[0]; nor[i * 3 + 1] = g[1]; nor[i * 3 + 2] = g[2];
    }
    return nor;
  }

  /* ---------------------------------------------------------------- rigs */
  // Head (head-local, +z nose). hp tweaks proportions per cat.
  function headSpec(hp) {
    const cw = hp.cheek, jw = hp.jowl, mz = hp.muzzle;
    const E = (c, r, k, rot) => ({ t: 'e', c, r, k, rot });
    const Cn = (a, b, ra, rb, k) => ({ t: 'c', a, b, ra, rb, k });
    const list = [
      E([0, 0.0085, -0.0135], [0.0335, 0.0305, 0.0368], 0),                 // cranium
      E([0, 0.0115, 0.0085], [0.0262, 0.0195, 0.0245], 0.013),              // forehead / brow mass
      E([0, -0.0010, 0.0090], [0.0262, 0.0200, 0.0255], 0.012),             // mid face fill under the eyes
      E([0.0212 * cw, -0.0085, 0.0010], [0.0212 * cw, 0.0205, 0.0245], 0.015), // cheeks
      E([-0.0212 * cw, -0.0085, 0.0010], [0.0212 * cw, 0.0205, 0.0245], 0.015),
      E([0.0160 * cw, -0.0192, -0.0010], [0.0165 * jw, 0.0125, 0.0200], 0.012), // jowl ruff
      E([-0.0160 * cw, -0.0192, -0.0010], [0.0165 * jw, 0.0125, 0.0200], 0.012),
      E([0, -0.0128, 0.0255 * mz], [0.0150, 0.0124, 0.0145], 0.010),        // muzzle base
      E([0.0083, -0.0170, 0.0360 * mz], [0.0100, 0.0080, 0.0086], 0.0055),  // whisker pads
      E([-0.0083, -0.0170, 0.0360 * mz], [0.0100, 0.0080, 0.0086], 0.0055),
      E([0, -0.0283, 0.0290 * mz], [0.0083, 0.0060, 0.0082], 0.0065),       // chin
      Cn([0, 0.0120, 0.0300], [0, -0.0040, 0.0418 * mz], 0.0092, 0.0055, 0.0085), // nose bridge
      E([0.0118, 0.0150, 0.0265], [0.0085, 0.0050, 0.0080], 0.006),        // brow pads over the eyes
      E([-0.0118, 0.0150, 0.0265], [0.0085, 0.0050, 0.0080], 0.006),
    ];
    const eyeW = hp.eyeW, eyeH = hp.eyeH;
    const eyes = [1, -1].map((s) => ({ side: s, c: [s * 0.0176, 0.0045, 0.0270], axis: new Vector3(s * 0.30, 0.03, 1).normalize(), w: eyeW, h: eyeH, slant: 0.17, lidDrop: hp.lidDrop || 0 }));
    const ears = [1, -1].map((s) => ({ side: s, base: [s * 0.0232, 0.0255, -0.0080], tilt: s * (hp.earTilt ?? 0.44), turn: s * 0.40, lean: -0.10, W: hp.earW, H: hp.earH }));
    return { list, eyes, ears, nose: { y: -0.0040, w: 0.0050 }, scale: hp.scale };
  }

  // helpers to build legs
  function legPrims(out, side, pts, radii, paw, k, tag) {
    for (let i = 0; i < pts.length - 1; i++) out.push(cone(pts[i], pts[i + 1], radii[i], radii[i + 1], i === 0 ? k : 0.010, tag, { seg: i }));
    if (paw) {
      const q = new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), V(paw.dir).normalize());
      out.push(ell(paw.c, paw.r, q, 0.009, 'paw', { front: paw.front, side, dir: V(paw.dir).normalize().toArray() }));
      if (paw.toes) {
        const dir = V(paw.dir).normalize(), sideV = new Vector3().crossVectors(new Vector3(0, 1, 0), dir).normalize();
        for (const o of [-1.5, -0.5, 0.5, 1.5]) {
          const c = V(paw.c).addScaledVector(dir, paw.r[2] * 0.72 - Math.abs(o) * 0.0022).addScaledVector(sideV, o * paw.r[0] * 0.42);
          c.y = Math.max(c.y - paw.r[1] * 0.18, paw.r[1] * 0.62);
          out.push(ell(c.toArray(), [paw.r[0] * 0.30, paw.r[1] * 0.62, paw.r[0] * 0.36], q, 0.0042, 'paw', { front: paw.front, side, dir: dir.toArray() }));
        }
      }
    }
  }

  const POSES = {
    // ---------------------------------------------------- tabby: sphinx loaf
    loaf() {
      const spine = [
        { p: [0, 0.086, -0.196], w: 0.040, hu: 0.040, hd: 0.062 },
        { p: [0, 0.108, -0.158], w: 0.058, hu: 0.051, hd: 0.100 },
        { p: [0, 0.119, -0.100], w: 0.061, hu: 0.055, hd: 0.114 },
        { p: [0, 0.123, -0.038], w: 0.062, hu: 0.057, hd: 0.119 },
        { p: [0, 0.127, 0.020], w: 0.058, hu: 0.057, hd: 0.121 },
        { p: [0, 0.135, 0.066], w: 0.053, hu: 0.053, hd: 0.112 },
        { p: [0, 0.148, 0.098], w: 0.043, hu: 0.044, hd: 0.080 },
      ];
      const head = { pos: [0, 0.214, 0.160], yaw: 0.36, pitch: -0.14, roll: 0.05 };
      const legs = [];
      for (const s of [1, -1]) {
        legs.push({ tag: 'fleg', side: s, pts: [[s * 0.030, 0.108, 0.066], [s * 0.043, 0.038, 0.034], [s * 0.035, 0.021, 0.118]], radii: [0.029, 0.0185, 0.0150],
          paw: { c: [s * 0.0325, 0.0140, 0.1435], r: [0.0178, 0.0122, 0.0228], dir: [s * -0.05, 0, 1], toes: true, front: true } });
        legs.push({ tag: 'hleg', side: s, pts: [[s * 0.036, 0.092, -0.158], [s * 0.060, 0.058, -0.090], [s * 0.052, 0.024, -0.178], [s * 0.042, 0.016, -0.104]], radii: [0.026, 0.0215, 0.0135, 0.0122],
          paw: { c: [s * 0.041, 0.0128, -0.098], r: [0.0160, 0.0110, 0.0235], dir: [0, 0, 1], toes: true, front: false },
          thigh: { c: [s * 0.050, 0.080, -0.128], r: [0.032, 0.046, 0.058], dir: [s * 0.25, -0.35, 0.9] } });
      }
      const tail = [[0, 0.074, -0.200], [0.012, 0.040, -0.240], [0.055, 0.017, -0.246], [0.092, 0.0135, -0.205], [0.106, 0.0130, -0.140], [0.103, 0.0128, -0.075], [0.090, 0.0128, -0.020]];
      const extras = [
        { c: [0, 0.085, 0.085], r: [0.040, 0.050, 0.030], k: 0.03, tag: 'chest' },                // chest front (bib)
        { c: [0, 0.030, -0.060], r: [0.058, 0.032, 0.125], k: 0.03, tag: 'belly' },              // belly on the counter
        { c: [0.028, 0.165, 0.060], r: [0.017, 0.014, 0.028], k: 0.022, tag: 'torso' },          // shoulder blades
        { c: [-0.028, 0.165, 0.060], r: [0.017, 0.014, 0.028], k: 0.022, tag: 'torso' },
      ];
      return { spine, head, legs, tail, tailR: [0.0160, 0.0092], extras, neck: { base: [0, 0.150, 0.092], rb: 0.040, rh: 0.031 }, throat: { a: [0, 0.098, 0.098], ra: 0.038 } };
    },
    // ----------------------------------- black (green collar): reclining on one hip, chest up
    recline(mirror = 1, chub = 1) {
      const s = mirror; // dorsal side points to -s*x ; legs reach +s*x
      const spine = [
        { p: [-s * 0.010, 0.066, -0.205], up: [-s, 0.25, 0], w: 0.050 * chub, hu: 0.040, hd: 0.055 * chub },
        { p: [-s * 0.002, 0.076, -0.160], up: [-s, 0.20, 0], w: 0.066 * chub, hu: 0.050, hd: 0.078 * chub },
        { p: [s * 0.004, 0.080, -0.098], up: [-s, 0.22, 0], w: 0.069 * chub, hu: 0.054, hd: 0.088 * chub },
        { p: [s * 0.006, 0.083, -0.036], up: [-s, 0.30, 0], w: 0.068 * chub, hu: 0.056, hd: 0.092 * chub },
        { p: [s * 0.004, 0.090, 0.024], up: [-s * 0.85, 0.52, 0], w: 0.062 * chub, hu: 0.056, hd: 0.092 * chub },
        { p: [-s * 0.002, 0.104, 0.070], up: [-s * 0.55, 0.84, 0.05], w: 0.054 * chub, hu: 0.052, hd: 0.085 * chub },
        { p: [-s * 0.004, 0.124, 0.102], up: [-s * 0.30, 0.95, 0.05], w: 0.044 * chub, hu: 0.044, hd: 0.064 * chub },
      ];
      const head = { pos: [s * 0.010, 0.178, 0.150], yaw: s * 0.30, pitch: -0.12, roll: -s * 0.06 };
      const legs = [];
      // front legs stretched forward, one slightly ahead of the other
      legs.push({ tag: 'fleg', side: s, pts: [[s * 0.026, 0.085, 0.074], [s * 0.050, 0.036, 0.090], [s * 0.060, 0.020, 0.176]], radii: [0.027, 0.0180, 0.0148],
        paw: { c: [s * 0.062, 0.0135, 0.2015], r: [0.0175, 0.0120, 0.0225], dir: [s * 0.12, 0, 1], toes: true, front: true } });
      legs.push({ tag: 'fleg', side: -s, pts: [[-s * 0.020, 0.070, 0.074], [-s * 0.004, 0.026, 0.098], [s * 0.012, 0.018, 0.168]], radii: [0.026, 0.0175, 0.0145],
        paw: { c: [s * 0.016, 0.0130, 0.1930], r: [0.0172, 0.0118, 0.0222], dir: [s * 0.08, 0, 1], toes: true, front: true } });
      // hind legs: upper one bent forward, lower one stretched out sideways
      legs.push({ tag: 'hleg', side: s, pts: [[s * 0.012, 0.110, -0.150], [s * 0.074, 0.098, -0.090], [s * 0.078, 0.060, -0.180], [s * 0.120, 0.040, -0.128]], radii: [0.027, 0.021, 0.0135, 0.0120],
        paw: { c: [s * 0.130, 0.036, -0.112], r: [0.0158, 0.0110, 0.0230], dir: [s * 0.55, -0.15, 0.82], toes: true, front: false },
        thigh: { c: [s * 0.040, 0.105, -0.122], r: [0.034, 0.044, 0.058], dir: [s * 0.75, 0.0, 0.66] } });
      legs.push({ tag: 'hleg', side: -s, pts: [[s * 0.004, 0.040, -0.150], [s * 0.070, 0.026, -0.112], [s * 0.098, 0.018, -0.190], [s * 0.150, 0.014, -0.168]], radii: [0.026, 0.020, 0.0130, 0.0118],
        paw: { c: [s * 0.165, 0.0128, -0.160], r: [0.0155, 0.0108, 0.0228], dir: [s * 0.9, 0, 0.42], toes: true, front: false },
        thigh: { c: [s * 0.030, 0.042, -0.130], r: [0.033, 0.036, 0.056], dir: [s * 0.8, 0, 0.6] } });
      const tail = [[-s * 0.012, 0.060, -0.205], [-s * 0.020, 0.030, -0.250], [s * 0.000, 0.0145, -0.300], [s * 0.040, 0.0135, -0.338], [s * 0.095, 0.0130, -0.352], [s * 0.150, 0.0128, -0.340]];
      const extras = [
        { c: [s * 0.040, 0.048, -0.050], r: [0.050 * chub, 0.042 * chub, 0.105], k: 0.035, tag: 'belly' },
        { c: [s * 0.018, 0.070, 0.080], r: [0.034, 0.046, 0.030], k: 0.028, tag: 'chest' },
      ];
      return { spine, head, legs, tail, tailR: [0.0165, 0.0090], extras, neck: { base: [-s * 0.004, 0.128, 0.094], rb: 0.041 * Math.sqrt(chub), rh: 0.032 }, throat: { a: [s * 0.010, 0.088, 0.096], ra: 0.036 } };
    },
    // ------------------------------- chubby black (orange collar): full side sprawl on the table
    sprawl(mirror = 1, chub = 1.18) {
      const s = mirror; // dorsal toward -s*x, belly & legs toward +s*x, left side down when s=1
      const spine = [
        { p: [-s * 0.030, 0.060, -0.200], up: [-s, 0.06, 0], w: 0.052 * chub, hu: 0.040, hd: 0.056 * chub },
        { p: [-s * 0.024, 0.070, -0.158], up: [-s, 0.05, 0], w: 0.068 * chub, hu: 0.050, hd: 0.082 * chub },
        { p: [-s * 0.018, 0.074, -0.096], up: [-s, 0.04, 0], w: 0.070 * chub, hu: 0.054, hd: 0.095 * chub },
        { p: [-s * 0.016, 0.075, -0.032], up: [-s, 0.05, 0], w: 0.070 * chub, hu: 0.056, hd: 0.100 * chub },
        { p: [-s * 0.018, 0.075, 0.028], up: [-s, 0.08, 0], w: 0.064 * chub, hu: 0.056, hd: 0.094 * chub },
        { p: [-s * 0.024, 0.076, 0.076], up: [-s, 0.14, 0], w: 0.055 * chub, hu: 0.052, hd: 0.080 * chub },
        { p: [-s * 0.028, 0.080, 0.110], up: [-s * 0.95, 0.30, 0], w: 0.045 * chub, hu: 0.045, hd: 0.060 * chub },
      ];
      const head = { pos: [-s * 0.016, 0.118, 0.170], yaw: s * 0.30, pitch: -0.08, roll: -s * 0.38 };
      const legs = [];
      // front legs reaching forward and toward the belly side
      legs.push({ tag: 'fleg', side: s, pts: [[-s * 0.004, 0.096, 0.082], [s * 0.036, 0.066, 0.124], [s * 0.066, 0.040, 0.192]], radii: [0.027, 0.0182, 0.0150],
        paw: { c: [s * 0.076, 0.034, 0.214], r: [0.0178, 0.0122, 0.0228], dir: [s * 0.40, -0.30, 0.86], toes: true, front: true } });
      legs.push({ tag: 'fleg', side: -s, pts: [[-s * 0.010, 0.040, 0.080], [s * 0.040, 0.020, 0.112], [s * 0.084, 0.017, 0.180]], radii: [0.027, 0.0180, 0.0148],
        paw: { c: [s * 0.098, 0.0135, 0.200], r: [0.0175, 0.0120, 0.0226], dir: [s * 0.55, 0, 0.83], toes: true, front: true } });
      // hind legs stretched out behind / to the side
      legs.push({ tag: 'hleg', side: s, pts: [[-s * 0.010, 0.104, -0.148], [s * 0.060, 0.112, -0.118], [s * 0.090, 0.090, -0.200], [s * 0.140, 0.074, -0.236]], radii: [0.028, 0.0215, 0.0138, 0.0122],
        paw: { c: [s * 0.154, 0.070, -0.246], r: [0.0160, 0.0112, 0.0236], dir: [s * 0.8, -0.2, -0.56], toes: true, front: false },
        thigh: { c: [s * 0.020, 0.104, -0.132], r: [0.036, 0.046, 0.060], dir: [s * 0.9, 0.1, 0.4] } });
      legs.push({ tag: 'hleg', side: -s, pts: [[-s * 0.010, 0.036, -0.148], [s * 0.060, 0.024, -0.110], [s * 0.082, 0.018, -0.196], [s * 0.130, 0.0145, -0.236]], radii: [0.027, 0.021, 0.0135, 0.0120],
        paw: { c: [s * 0.146, 0.0128, -0.246], r: [0.0158, 0.0110, 0.0232], dir: [s * 0.72, 0, -0.69], toes: true, front: false },
        thigh: { c: [s * 0.020, 0.040, -0.130], r: [0.034, 0.036, 0.058], dir: [s * 0.9, 0, 0.4] } });
      const tail = [[-s * 0.034, 0.056, -0.200], [-s * 0.042, 0.030, -0.245], [-s * 0.034, 0.0145, -0.300], [-s * 0.006, 0.0135, -0.348], [s * 0.040, 0.0130, -0.378], [s * 0.096, 0.0128, -0.386]];
      const extras = [
        { c: [s * 0.050, 0.054, -0.060], r: [0.054 * chub, 0.050 * chub, 0.115], k: 0.04, tag: 'belly' },
        { c: [s * 0.010, 0.066, 0.090], r: [0.040, 0.040, 0.030], k: 0.03, tag: 'chest' },
      ];
      return { spine, head, legs, tail, tailR: [0.0170, 0.0092], extras, neck: { base: [-s * 0.028, 0.084, 0.104], rb: 0.044 * Math.sqrt(chub), rh: 0.034 }, throat: { a: [s * 0.006, 0.066, 0.104], ra: 0.038 } };
    },
  };

  function headMatrix(h, scale) {
    const q = new Quaternion().setFromEuler(new Euler(h.pitch, h.yaw, h.roll, 'YXZ'));
    const M = new Matrix4().compose(V(h.pos), q, new Vector3(scale, scale, scale));
    return { q, M, inv: M.clone().invert() };
  }

  function assemble(cfg) {
    const rig = POSES[cfg.pose](cfg.mirror, cfg.chub);
    const hs = headSpec(cfg.head);
    const S = hs.scale;
    const { q: hq, M: HM, inv: HI } = headMatrix(rig.head, S);
    const tp = (p) => V(p).applyMatrix4(HM).toArray();
    const prims = [], subs = [];
    // torso
    prims.push(sweep(rig.spine, 128, 0.02, 'torso'));
    // neck + throat into the head
    const headBase = tp([0, -0.010, -0.028]), chinBack = tp([0, -0.030, 0.000]);
    prims.push(cone(rig.neck.base, headBase, rig.neck.rb, rig.neck.rh * S, 0.024, 'neck'));
    prims.push(cone(rig.throat.a, chinBack, rig.throat.ra, 0.020 * S, 0.022, 'throat'));
    for (const e of rig.extras) prims.push(ell(e.c, e.r, null, e.k, e.tag));
    // head
    for (const h of hs.list) {
      if (h.t === 'e') prims.push(ell(tp(h.c), h.r.map((v) => v * S), hq, h.k * S || 0.012, 'head'));
      else prims.push(cone(tp(h.a), tp(h.b), h.ra * S, h.rb * S, h.k * S, 'head'));
    }
    // legs
    for (const L of rig.legs) {
      if (L.thigh) {
        const q = new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), V(L.thigh.dir).normalize());
        prims.push(ell(L.thigh.c, L.thigh.r, q, 0.026, 'thigh', { side: L.side }));
      }
      legPrims(prims, L.side, L.pts, L.radii, L.paw, 0.022, L.tag);
    }
    // tail
    const tailCtrl = rig.tail.map((p, i, a) => { const t = i / (a.length - 1); const r = lerp(rig.tailR[0], rig.tailR[1], Math.pow(t, 0.9)); return { p, w: r, hu: r, hd: r }; });
    const tailP = sweep(tailCtrl, 96, 0.016, 'tail');
    tailP.capStart = false;
    prims.push(tailP);
    const ground = { y: 0.0035, k: 0.010 };
    const fUnion = makeSDF(prims, [], ground);
    // eyes: find surface along the eye axis, carve almond sockets
    const eyes = hs.eyes.map((e) => {
      const axisC = e.axis.clone().applyQuaternion(hq).normalize();
      const cC = V(tp(e.c));
      const start = cC.clone().addScaledVector(axisC, 0.03);
      const hit = rayHit(fUnion, start.toArray(), axisC.clone().negate().toArray(), 0.05, 0.0004) || cC.toArray();
      const surf = V(hit);
      // eye frame: x across (with slant), y up, z axis
      const upH = new Vector3(0, 1, 0).applyQuaternion(hq);
      const ex = new Vector3().crossVectors(upH, axisC).normalize(); // points to the cat's left
      const ey = new Vector3().crossVectors(axisC, ex).normalize();
      const sl = e.slant * e.side; // outer corner up
      const exS = ex.clone().multiplyScalar(Math.cos(sl)).addScaledVector(ey, Math.sin(sl));
      const eyS = ey.clone().multiplyScalar(Math.cos(sl)).addScaledVector(ex, -Math.sin(sl));
      const qE = new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(exS, eyS, axisC));
      const wS = e.w * S, hS = e.h * S;
      const cc = surf.clone().addScaledVector(axisC, -0.0010 * S).addScaledVector(eyS, -e.lidDrop * S);
      subs.push(eyeCarve(cc.toArray(), qE, wS, hS, 0.0058 * S, 0.0015 * S, 0.30));
      return { side: e.side, surf, axis: axisC, ex: exS, ey: eyS, q: qE, w: wS, h: hS, c: cc, R: 0.0102 * S, pinch: 0.30 };
    });
    const f = makeSDF(prims, subs, ground);
    return { rig, hs, S, hq, HM, HI, prims, subs, f, fUnion, eyes, ground };
  }

  /* --------------------------------------------------------------- paint */
  const PAL = {
    tabby: {
      ground: C('#7f6d57'), groundLight: C('#a8957a'), groundWarm: C('#8e7454'), stripe: C('#221c16'), dorsal: C('#3a3026'),
      white: C('#ece8de'), cream: C('#d8cdb6'), bridge: C('#94704f'), liner: C('#1a1714'), earIn: C('#cf9f97'), earHair: C('#e6ddca'),
      tipLight: C('#c9b796'),
    },
    black: {
      ground: C('#161412'), groundLight: C('#1f1b17'), stripe: C('#0e0d0c'), white: C('#141210'), liner: C('#060606'), earIn: C('#3a2f2e'), earHair: C('#2a2420'),
      tipLight: C('#3b2f25'),
    },
  };
  // side outputs of paint(): hair flow direction (cat-local, unit) and hair length (m)
  const OUT = { fx: 0, fy: 0, fz: -1, len: 0 };

  function painter(A, kind) {
    const P = PAL[kind];
    const tabby = kind === 'tabby';
    const prims = A.prims;
    const tags = prims.map((p) => p.tag);
    const torso = prims.find((p) => p.tag === 'torso' && p.kind === K_SWEEP);
    const tailP = prims.find((p) => p.tag === 'tail');
    const eyes = A.eyes;
    const HI = A.HI.elements, HQ = A.hq;
    const toHead = (x, y, z, o) => { o[0] = HI[0] * x + HI[4] * y + HI[8] * z + HI[12]; o[1] = HI[1] * x + HI[5] * y + HI[9] * z + HI[13]; o[2] = HI[2] * x + HI[6] * y + HI[10] * z + HI[14]; return o; };
    const rotH = (v) => { // head-local direction → cat-local
      const qx = HQ.x, qy = HQ.y, qz = HQ.z, qw = HQ.w;
      const tx = 2 * (qy * v[2] - qz * v[1]), ty = 2 * (qz * v[0] - qx * v[2]), tz = 2 * (qx * v[1] - qy * v[0]);
      return [v[0] + qw * tx + (qy * tz - qz * ty), v[1] + qw * ty + (qz * tx - qx * tz), v[2] + qw * tz + (qx * ty - qy * tx)];
    };
    const HB = [0, 0, 0];
    // eye-local helper (lemon shaped like the carve)
    const ER = [9, 0];
    const eyeRho = (x, y, z) => {
      let best = 9, lzb = 0;
      for (const e of eyes) {
        const dx = x - e.c.x, dy = y - e.c.y, dz = z - e.c.z;
        const lx = dx * e.ex.x + dy * e.ex.y + dz * e.ex.z, ly = dx * e.ey.x + dy * e.ey.y + dz * e.ey.z, lz = dx * e.axis.x + dy * e.axis.y + dz * e.axis.z;
        const xn = lx / e.w, hh = e.h * Math.max(0.08, 1 - e.pinch * xn * xn) * (ly > 0 ? 0.97 : 1.05);
        const r = Math.hypot(xn, ly / hh);
        if (r < best) { best = r; lzb = lz; }
      }
      ER[0] = best; ER[1] = lzb;
      return ER;
    };
    // tabby face markings in head-local
    const M_LINES = [
      [[0, 0.0165], [0, 0.0345]],
      [[0.0058, 0.0178], [0.0050, 0.0350]],
      [[-0.0058, 0.0178], [-0.0050, 0.0350]],
      [[0.0125, 0.0148], [0.0108, 0.0235], [0.0078, 0.0320]],
      [[-0.0125, 0.0148], [-0.0108, 0.0235], [-0.0078, 0.0320]],
    ];
    const CROWN = [0, 0.0058, -0.0058, 0.0118, -0.0118];
    const CHEEK1 = [[0.0215, 0.0048], [0.0080, 0.0018], [-0.0050, -0.0030], [-0.0160, -0.0042]]; // (z, y) from outer eye corner backwards
    const CHEEK2 = [[0.0160, -0.0075], [0.0050, -0.0105], [-0.0065, -0.0140], [-0.0150, -0.0130]];

    function headColor(x, y, z, hx, hy, hz) {
      const n1 = fbm(x * 260, y * 260, z * 260, 3), tick = vnoise(x * 1500, y * 1500, z * 1500, 17);
      if (!tabby) return mixc(P.ground, P.groundLight, n1 * 0.55 + tick * 0.25);
      let c = mixc(P.ground, P.groundWarm, 0.3 + n1 * 0.35);
      c = mixc(c, P.groundLight, tick * 0.25 + sstep(-0.004, -0.014, hy) * 0.18);
      const ax = Math.abs(hx);
      // white muzzle (whisker pads + chin) and throat
      const mz = Math.hypot(hx / 0.0190, (hy + 0.0222) / 0.0108);
      let white = (1 - sstep(0.86, 1.10, mz)) * sstep(0.014, 0.028, hz);
      white = Math.max(white, sstep(-0.0215, -0.0275, hy) * sstep(-0.040, -0.010, hz));
      white *= 1 - (1 - sstep(0.0040, 0.0068, ax)) * sstep(-0.0150, -0.0118, hy) * 0.95; // keep the nose bridge/top lip line coloured
      // dark tabby lines
      let dark = 0;
      if (hz > -0.004 && hy > 0.011) {
        for (const L of M_LINES) dark = Math.max(dark, 1 - sstep(0.0007, 0.0017, dPoly(hx, hy, L)));
        dark *= sstep(-0.004, 0.012, hz);
      }
      if (hy > 0.004) {
        for (const cx of CROWN) {
          const wob = 0.0010 * Math.sin(hz * 260 + cx * 400);
          const d = Math.abs(hx - cx * (1 + Math.max(0, -hz) * 7) - wob);
          dark = Math.max(dark, (1 - sstep(0.0010, 0.0022, d)) * sstep(0.013, -0.004, hz) * sstep(0.006, 0.020, hy));
        }
      }
      if (ax > 0.013 && hz < 0.028) {
        const d1 = dPoly(hz, hy, CHEEK1), d2 = dPoly(hz, hy, CHEEK2);
        dark = Math.max(dark, (1 - sstep(0.0009, 0.0020, d1)) * sstep(0.013, 0.021, ax), (1 - sstep(0.0008, 0.0018, d2)) * sstep(0.016, 0.024, ax) * 0.9);
      }
      // warm nose bridge
      const br = (1 - sstep(0.0032, 0.0060, ax)) * sstep(0.024, 0.034, hz) * sstep(-0.013, -0.007, hy) * (1 - sstep(0.008, 0.016, hy));
      c = mixc(c, P.bridge, br * 0.9);
      // cream rims around the eyes
      const er = eyeRho(x, y, z);
      c = mixc(c, P.cream, sstep(1.62, 1.22, er[0]) * sstep(-0.006, -0.001, er[1]) * 0.8);
      c = mixc(c, P.stripe, dark * 0.92);
      c = mixc(c, P.white, clamp(white, 0, 1));
      c = mixc(c, P.dorsal, sstep(0.020, 0.032, hy) * sstep(0.0, -0.03, hz) * 0.35);
      return c;
    }

    function regionColor(i, x, y, z) {
      if (i < 0) return P.ground;
      const p = prims[i], tag = p.tag;
      const n1 = fbm(x * 90, y * 90, z * 90, 1), tick = vnoise(x * 1400, y * 1400, z * 1400, 5);
      if (tag === 'head') { toHead(x, y, z, HB); return headColor(x, y, z, HB[0], HB[1], HB[2]); }
      if (!tabby) {
        let c = mixc(P.ground, P.groundLight, n1 * 0.6 + tick * 0.3);
        if (tag === 'paw') c = mixc(c, C('#100f0e'), 0.4);
        return c;
      }
      // ---------------- tabby body
      let base = mixc(P.ground, P.groundWarm, 0.25 + n1 * 0.4);
      base = mixc(base, P.groundLight, tick * 0.28);
      if (tag === 'torso' || tag === 'belly' || tag === 'chest' || tag === 'neck' || tag === 'throat' || tag === 'thigh') {
        evalPrim(torso, x, y, z, true);
        const sArc = LOC.s, ph = Math.abs(LOC.phi), tt = LOC.t;
        const warp = (fbm(x * 30, y * 30, z * 30, 9) - 0.5) * 0.75 + 0.16 * Math.sin(ph * 2.0);
        const v = 0.5 + 0.5 * Math.cos((sArc / 0.0205 + warp) * Math.PI * 2);
        let stripe = sstep(0.66, 0.90, v) * (1 - sstep(1.65, 2.25, ph));
        const dash = sstep(0.32, 0.58, vnoise(sArc * 95, ph * 3.2, 0.5, 4));
        stripe *= 1 - sstep(1.05, 1.75, ph) * (1 - dash) * 0.85;
        const dorsal = 1 - sstep(0.13, 0.32, ph + (n1 - 0.5) * 0.1);
        let c = mixc(base, P.dorsal, dorsal * 0.6);
        c = mixc(c, P.stripe, Math.max(stripe * 0.88, dorsal * 0.5));
        const ventral = sstep(1.95, 2.6, ph);
        c = mixc(c, P.groundLight, ventral * 0.45);
        let bib = 0;
        if (tag === 'throat') bib = 1;
        else if (tag === 'neck') bib = sstep(1.65, 2.25, ph);
        else if (tag === 'chest') { evalPrim(p, x, y, z, true); bib = sstep(-0.35, 0.25, LOC.lz) * (1 - sstep(0.55, 0.85, Math.abs(LOC.lx))); }
        else bib = sstep(0.82, 0.93, tt) * sstep(2.05, 2.55, ph);
        c = mixc(c, P.white, clamp(bib, 0, 1) * 0.97);
        if (tag === 'belly') c = mixc(c, P.groundLight, 0.35);
        if (tag === 'thigh') {
          evalPrim(p, x, y, z, true);
          const bar = sstep(0.64, 0.86, 0.5 + 0.5 * Math.cos((LOC.lz * 2.4 + LOC.ly * 0.8 + warp * 0.35) * Math.PI));
          c = mixc(base, P.stripe, bar * 0.82);
          c = mixc(c, P.groundLight, sstep(0.2, -0.6, LOC.ly) * 0.35);
        }
        return c;
      }
      if (tag === 'fleg' || tag === 'hleg' || tag === 'paw') {
        if (tag === 'paw') return mixc(P.white, P.cream, tick * 0.3);
        evalPrim(p, x, y, z, true);
        let sock = 0;
        if (tag === 'fleg') sock = p.seg === 1 ? sstep(0.70, 0.88, LOC.t) : 0;
        if (tag === 'hleg') sock = p.seg === 2 ? sstep(0.15, 0.45, LOC.t) : 0;
        const ph = (y + z * 0.12 + x * 0.05) / 0.019 + (n1 - 0.5) * 1.0;
        const bar = sstep(0.62, 0.86, 0.5 + 0.5 * Math.cos(ph * Math.PI * 2));
        let c = mixc(base, P.stripe, bar * 0.78 * (p.seg === 0 ? 0.55 : 1));
        c = mixc(c, P.groundLight, 0.2);
        return mixc(c, P.white, sock);
      }
      if (tag === 'tail') {
        evalPrim(tailP, x, y, z, true);
        const sArc = LOC.s, len = tailP.total;
        const ring = sstep(0.56, 0.80, 0.5 + 0.5 * Math.cos((sArc / 0.031) * Math.PI * 2 + 0.6 + (n1 - 0.5) * 0.8));
        const tip = sstep(len - 0.046, len - 0.032, sArc);
        const under = sstep(1.9, 2.6, Math.abs(LOC.phi));
        let c = mixc(base, P.dorsal, 0.25);
        return mixc(c, P.stripe, Math.max(ring * (1 - under * 0.55), tip) * 0.9);
      }
      return base;
    }

    function regionFlow(i, x, y, z) {
      const p = i >= 0 ? prims[i] : null, tag = p ? p.tag : 'torso';
      let fx = 0, fy = 0, fz = -1, len = 0.0045;
      if (tag === 'head') {
        toHead(x, y, z, HB);
        const hx = HB[0], hy = HB[1] + 0.006, hz = HB[2] - 0.048;
        const r = Math.hypot(hx, hy, hz) || 1;
        const v = rotH([hx / r, hy / r, hz / r]);
        fx = v[0]; fy = v[1]; fz = v[2];
        len = lerp(0.0011, 0.0032, sstep(0.012, 0.05, r)) + sstep(0.022, 0.034, Math.abs(HB[0])) * sstep(0.0, -0.02, HB[1]) * 0.0016;
      } else if (tag === 'tail') {
        evalPrim(tailP, x, y, z, true); fx = LOC.tx; fy = LOC.ty; fz = LOC.tz; len = 0.0050;
      } else if (tag === 'fleg' || tag === 'hleg') {
        const l = p.len || 1; fx = p.bax / l; fy = p.bay / l; fz = p.baz / l; len = 0.0032;
      } else if (tag === 'paw') {
        const d = p.dir || [0, 0, 1]; fx = d[0]; fy = d[1] - 0.3; fz = d[2]; len = 0.0020;
      } else {
        evalPrim(torso, x, y, z, true);
        const side = sstep(0.25, 1.3, Math.abs(LOC.phi)) * 0.5;
        fx = -LOC.tx - LOC.ux * side; fy = -LOC.ty - LOC.uy * side; fz = -LOC.tz - LOC.uz * side;
        len = tag === 'neck' || tag === 'throat' || tag === 'chest' ? 0.0056 : tag === 'belly' ? 0.0050 : 0.0046;
      }
      const l = Math.hypot(fx, fy, fz) || 1;
      OUT.fx = fx / l; OUT.fy = fy / l; OUT.fz = fz / l; OUT.len = len * (tabby ? 1 : 1.12);
    }

    return function paint(x, y, z) {
      let d1 = 1e9, d2 = 1e9, i1 = -1, i2 = -1;
      for (let i = 0; i < prims.length; i++) {
        const p = prims[i], bb = p.bb;
        if (x < bb[0] || y < bb[1] || z < bb[2] || x > bb[3] || y > bb[4] || z > bb[5]) continue;
        const d = evalPrim(p, x, y, z, false);
        if (d < d1) { d2 = d1; i2 = i1; d1 = d; i1 = i; } else if (d < d2) { d2 = d; i2 = i; }
      }
      let c = regionColor(i1, x, y, z);
      if (i2 >= 0 && tags[i2] !== tags[i1] && d2 - d1 < 0.012) {
        const c2 = regionColor(i2, x, y, z);
        c = mixc(c2, c, 0.5 + 0.5 * sstep(0, 0.012, d2 - d1));
      }
      regionFlow(i1, x, y, z);
      // eyelid rims: dark liner; no hair strands right at the lids
      const er = eyeRho(x, y, z);
      if (er[0] < 2) {
        const lid = sstep(-0.0075, -0.0015, er[1]);
        c = mixc(c, P.liner, sstep(1.24, 1.03, er[0]) * lid);
        OUT.len *= 1 - sstep(1.7, 1.15, er[0]) * lid;
      }
      return c;
    };
  }

  /* ------------------------------------------------------- accessory meshes */
  function earGeometry(sp, kind) {
    const P = PAL[kind], tabby = kind === 'tabby';
    const nu = 18, nv = 16;
    const W = sp.W, H = sp.H, D = sp.D, T = sp.T;
    const half = (u) => W * 0.5 * Math.pow(Math.max(0, 1 - u), 0.80) * (1 + 0.20 * Math.sin(Math.PI * Math.min(u, 1)));
    const zBack = (u, v) => -D * Math.pow(Math.max(0, 1 - v * v), 0.75) * Math.pow(Math.max(0, 1 - u), 0.55) + sp.curl * u * u;
    const pos = [], col = [], idx = [];
    const outerC = tabby ? C('#7a6a55') : P.ground, edgeC = tabby ? C('#3a3128') : P.ground;
    const add = (x, y, z, c) => { pos.push(x, y, z); col.push(...c); return pos.length / 3 - 1; };
    const u0 = -0.16;
    const grid = (front) => {
      const base = pos.length / 3;
      for (let i = 0; i <= nu; i++) {
        const u = u0 + (1 - u0) * Math.pow(i / nu, 0.92);
        for (let j = 0; j <= nv; j++) {
          const v = -1 + (2 * j) / nv;
          const hw = half(u);
          if (!front) {
            const z = zBack(u, v);
            const edge = sstep(0.75, 1, Math.abs(v)) + sstep(0.82, 1, u);
            add(v * hw, u * H, z, mixc(outerC, edgeC, clamp(edge, 0, 1) * 0.7));
          } else {
            const rim = Math.min(0.0034, hw * 0.22) * (1 - u * 0.4);
            const th = T * (1 - 0.6 * Math.max(0, u));
            const deep = D * 0.42 * (1 - v * v) * Math.pow(Math.max(0, 1 - u), 2.2);
            const z = zBack(u, v) + th - deep * 0.15;
            const vv = Math.abs(v);
            const hair = clamp(sstep(0.45, 0.95, vv) * 0.85 + Math.pow(Math.max(0, 1 - u), 5) * 0.6 + (hash(i, j, 3, 2) - 0.5) * 0.3, 0, 1);
            let c = mixc(P.earIn, P.earHair, hair);
            if (!tabby) c = mixc(P.earIn, P.earHair, hair * 0.7);
            add(v * (hw - rim), u * H * 0.985, z, c);
          }
        }
      }
      for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
        const a = base + i * (nv + 1) + j, b = a + 1, c = a + nv + 1, d = c + 1;
        if (front) idx.push(a, b, d, a, d, c); else idx.push(a, d, b, a, c, d);
      }
      return base;
    };
    const back = grid(false), front = grid(true);
    // rim strips along the two side edges
    for (const j of [0, nv]) {
      for (let i = 0; i < nu; i++) {
        const a = back + i * (nv + 1) + j, b = back + (i + 1) * (nv + 1) + j, c = front + i * (nv + 1) + j, d = front + (i + 1) * (nv + 1) + j;
        if (j === 0) idx.push(a, c, b, b, c, d); else idx.push(a, b, c, b, d, c);
      }
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }

  function irisTexture(kind) {
    const N = 256, cv = document.createElement('canvas');
    cv.width = cv.height = N;
    const g = cv.getContext('2d');
    const img = g.createImageData(N, N);
    const tabby = kind === 'tabby';
    const inner = tabby ? [206, 186, 84] : [212, 196, 70], outer = tabby ? [150, 162, 72] : [150, 160, 52], limb = tabby ? [58, 66, 34] : [50, 54, 26];
    const pw = tabby ? 0.125 : 0.15, ph = 0.355;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const x = (i + 0.5) / N - 0.5, y = (j + 0.5) / N - 0.5;
      const r = Math.hypot(x, y) / 0.5, a = Math.atan2(y, x);
      const fib = 0.5 + 0.5 * Math.sin(a * 46 + Math.sin(a * 7) * 2.0) * (0.6 + 0.4 * Math.sin(r * 30));
      let c = r < 0.55 ? mixc(inner, outer, sstep(0.15, 0.55, r)) : mixc(outer, limb, sstep(0.72, 0.93, r));
      c = scalec(c, 0.86 + fib * 0.22);
      // pupil (vertical slit-oval) with soft edge
      const pr = Math.hypot(x / pw, y / ph);
      const pupil = 1 - sstep(0.88, 1.06, pr);
      c = mixc(c, [6, 7, 6], pupil);
      // darker ring right around the pupil
      c = mixc(c, scalec(c, 0.65), sstep(1.5, 1.05, pr) * (1 - pupil) * 0.6);
      const o = (j * N + i) * 4;
      img.data[o] = clamp(c[0], 0, 255); img.data[o + 1] = clamp(c[1], 0, 255); img.data[o + 2] = clamp(c[2], 0, 255); img.data[o + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    return t;
  }

  function eyeGeometry(R) {
    // spherical cap facing +z with planar uv; cornea slightly domed
    const g = new THREE.SphereGeometry(R, 40, 24, 0, Math.PI * 2, 0, Math.PI * 0.62);
    g.rotateX(Math.PI / 2);
    const p = g.attributes.position, uv = g.attributes.uv;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i);
      let z = p.getZ(i);
      const r = Math.hypot(x, y) / R;
      z += R * 0.10 * Math.max(0, 1 - r * r); // cornea bulge
      p.setZ(i, z);
      uv.setXY(i, 0.5 + x / (2 * R) / 0.92, 0.5 + y / (2 * R) / 0.92);
    }
    g.computeVertexNormals();
    return g;
  }

  function noseGeometry(kind, w) {
    const tabby = kind === 'tabby';
    const ctrl = [[0, 0.0026], [0.62, 0.0024], [0.98, 0.0012], [0.74, -0.0010], [0.30, -0.0032], [0, -0.0044]];
    const pts = [];
    for (const [x, y] of ctrl) pts.push(new Vector3(x * w, y, 0));
    for (let i = ctrl.length - 2; i > 0; i--) pts.push(new Vector3(-ctrl[i][0] * w, ctrl[i][1], 0));
    const curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal');
    const ring = curve.getSpacedPoints(48).slice(0, 48);
    const rings = 7, pos = [], col = [], idx = [];
    const cTop = tabby ? C('#cf8a7d') : C('#151313'), cRim = tabby ? C('#8f5a50') : C('#0c0b0b'), cNos = tabby ? C('#5c3532') : C('#050505');
    const cy = -0.0004;
    for (let r = 0; r <= rings; r++) {
      const t = r / rings;
      for (let i = 0; i < 48; i++) {
        const P = ring[i];
        const x = P.x * t, y = cy + (P.y - cy) * t;
        const z = 0.0017 * Math.pow(Math.max(0, 1 - t * t), 0.55) - 0.0003 * (1 - t) + 0.0006 * Math.max(0, y / 0.0026);
        let c = mixc(cTop, cRim, sstep(0.55, 1, t));
        if (tabby) {
          const nos = Math.max(1 - sstep(0.0007, 0.0013, Math.hypot((x - 0.0024) * 1.1, (y + 0.0011) * 1.6)), 1 - sstep(0.0007, 0.0013, Math.hypot((x + 0.0024) * 1.1, (y + 0.0011) * 1.6)));
          c = mixc(c, cNos, nos);
        }
        pos.push(x, y, z); col.push(...c);
      }
    }
    for (let r = 0; r < rings; r++) for (let i = 0; i < 48; i++) {
      const a = r * 48 + i, b = r * 48 + ((i + 1) % 48), c = a + 48, d = b + 48;
      idx.push(a, b, d, a, d, c);
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }

  // tapered tube along points
  function strandGeometry(points, r0, r1, radial = 4, seg = 18) {
    const curve = new THREE.CatmullRomCurve3(points.map(V));
    const g = new THREE.TubeGeometry(curve, seg, 1, radial, false);
    const p = g.attributes.position;
    for (let i = 0; i <= seg; i++) {
      const c = curve.getPointAt(i / seg), r = lerp(r0, r1, Math.pow(i / seg, 0.85));
      for (let j = 0; j <= radial; j++) {
        const k = i * (radial + 1) + j;
        p.setXYZ(k, c.x + (p.getX(k) - c.x) * r, c.y + (p.getY(k) - c.y) * r, c.z + (p.getZ(k) - c.z) * r);
      }
    }
    g.computeVertexNormals();
    return g;
  }

  function fabricTexture(hex) {
    const N = 256, cv = document.createElement('canvas');
    cv.width = cv.height = N;
    const g = cv.getContext('2d');
    const img = g.createImageData(N, N);
    const base = new Color(hex);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const w = 0.5 + 0.5 * Math.sin(i * Math.PI / 2) * Math.sin(j * Math.PI / 2);
      const n = hash(i, j, 1, 9) - 0.5;
      const k = 0.9 + w * 0.12 + n * 0.08;
      const o = (j * N + i) * 4;
      img.data[o] = clamp(base.r ** (1 / 2.2) * 255 * k, 0, 255);
      img.data[o + 1] = clamp(base.g ** (1 / 2.2) * 255 * k, 0, 255);
      img.data[o + 2] = clamp(base.b ** (1 / 2.2) * 255 * k, 0, 255);
      img.data[o + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(cv);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 4;
    return t;
  }

  function collarGroup(A, color, seed, petal = {}) {
    const f = A.fUnion;
    const rig = A.rig;
    const hb = V(A.rig.neck.base), head = V(new Vector3(0, -0.012, -0.030).applyMatrix4(A.HM).toArray());
    const axis = head.clone().sub(hb).normalize();
    const c = hb.clone().lerp(head, 0.60);
    const e1 = new Vector3().crossVectors(axis, new Vector3(0.3, 0.2, 1).normalize()).normalize();
    const e2 = new Vector3().crossVectors(axis, e1).normalize();
    const NS = 96, radius = [];
    for (let i = 0; i < NS; i++) {
      const a = (i / NS) * Math.PI * 2, d = e1.clone().multiplyScalar(Math.cos(a)).addScaledVector(e2, Math.sin(a));
      // march outward from the centre
      let r = 0.004;
      for (; r < 0.09; r += 0.0008) if (f(c.x + d.x * r, c.y + d.y * r, c.z + d.z * r) > 0) break;
      radius.push(r);
    }
    // smooth radii
    const rs = radius.map((_, i) => { let s = 0; for (let k = -3; k <= 3; k++) s += radius[(i + k + NS) % NS]; return s / 7; });
    const grp = new Group();
    const tex = fabricTexture(color);
    const mat = new THREE.MeshPhysicalNodeMaterial({ map: tex, roughness: 0.88, sheen: 0.6, sheenRoughness: 0.7, sheenColor: new Color(color).lerp(new Color('#ffffff'), 0.35), side: THREE.DoubleSide });
    // gathered band (scrunchie): fabric tube with irregular little folds
    const rndB = rng(seed * 7 + 1);
    const fold = (a) => (Math.sin(a * 37 + 2.8 * Math.sin(a * 3 + seed)) * 0.55 + Math.sin(a * 61 + 1.3 + 2.1 * Math.sin(a * 5 + seed)) * 0.30) * (0.65 + 0.35 * Math.sin(a * 7 + seed * 2));
    {
      const nA = 260, nB = 16, pos = [], uv = [], idx = [];
      const rb = 0.0056;
      for (let i = 0; i <= nA; i++) {
        const t = i / nA, a = t * Math.PI * 2;
        const ri = t * NS, R = lerp(rs[Math.floor(ri) % NS], rs[Math.ceil(ri) % NS], ri % 1) + rb * 0.45;
        const dir = e1.clone().multiplyScalar(Math.cos(a)).addScaledVector(e2, Math.sin(a));
        const tang = e1.clone().multiplyScalar(-Math.sin(a)).addScaledVector(e2, Math.cos(a));
        const fo = fold(a);
        for (let j = 0; j <= nB; j++) {
          const bb = (j / nB) * Math.PI * 2;
          const rr = rb * (1 + 0.085 * fo * (0.55 + 0.45 * Math.cos(bb)));
          const p = c.clone().addScaledVector(dir, R + Math.cos(bb) * rr * 0.85).addScaledVector(axis, Math.sin(bb) * rr * 1.35).addScaledVector(tang, 0.0009 * fo * Math.sin(bb * 2));
          pos.push(p.x, p.y, p.z); uv.push(t * 26, (j / nB) * 2);
        }
      }
      for (let i = 0; i < nA; i++) for (let j = 0; j < nB; j++) {
        const q = i * (nB + 1) + j, r = q + nB + 1;
        idx.push(q, r, q + 1, q + 1, r, r + 1);
      }
      const g = new BufferGeometry();
      g.setAttribute('position', new Float32BufferAttribute(pos, 3));
      g.setAttribute('uv', new Float32BufferAttribute(uv, 2));
      g.setIndex(idx); g.computeVertexNormals();
      grp.add(new Mesh(g, mat));
    }
    // pointed fabric petals fanning out from under the band (bigger toward the chest/front)
    const rnd = rng(seed);
    const front = new Vector3(0, -0.35, 1).normalize();
    const petalCount = 7;
    for (let k = 0; k < petalCount; k++) {
      const a0 = (k / petalCount) * Math.PI * 2 + (rnd() - 0.5) * 0.3 + seed * 0.37;
      const dir = e1.clone().multiplyScalar(Math.cos(a0)).addScaledVector(e2, Math.sin(a0));
      const tang = e1.clone().multiplyScalar(-Math.sin(a0)).addScaledVector(e2, Math.cos(a0));
      const facing = clamp(dir.dot(front) * 0.5 + 0.5, 0, 1);
      const ps = petal.scale || 1;
      const L = (0.020 + 0.016 * facing) * (0.88 + rnd() * 0.24) * ps, Wd = (0.018 + 0.008 * facing) * (0.9 + rnd() * 0.2) * Math.sqrt(ps);
      const ri = ((a0 / (Math.PI * 2)) % 1 + 1) % 1 * NS;
      const R = lerp(rs[Math.floor(ri) % NS], rs[Math.ceil(ri) % NS], ri % 1) + 0.0035;
      const droop0 = (petal.droop ?? 0.55) + rnd() * 0.25, bend = 0.55 + rnd() * 0.3, twist = (rnd() - 0.5) * 0.6;
      const nS = 14, nT = 12, pos = [], uv = [], idx = [];
      for (let i = 0; i <= nS; i++) {
        const sp = i / nS;
        const wdt = Wd * Math.pow(Math.sin(Math.PI * (0.18 + 0.82 * sp)), 0.85) * (1 - Math.pow(sp, 3.2)) + 0.0012 * (1 - sp);
        const dr = droop0 + bend * sp * sp;
        const along = dir.clone().multiplyScalar(Math.cos(dr)).addScaledVector(axis, -Math.sin(dr));
        const nrm = axis.clone().multiplyScalar(Math.cos(dr)).addScaledVector(dir, Math.sin(dr));
        const tw = tang.clone().multiplyScalar(Math.cos(twist * sp)).addScaledVector(nrm, Math.sin(twist * sp));
        for (let j = 0; j <= nT; j++) {
          const t = -1 + (2 * j) / nT;
          const crease = 0.0032 * (1 - Math.abs(t)) * Math.sin(Math.PI * sp) ; // folded along the midline
          const wave = 0.0016 * Math.sin(t * Math.PI * 1.7 + sp * 3 + k * 1.3) * (0.25 + sp) * Math.abs(t);
          const p = c.clone().addScaledVector(dir, R).addScaledVector(along, sp * L).addScaledVector(tw, t * wdt * 0.5).addScaledVector(nrm, crease + wave + 0.0008);
          pos.push(p.x, p.y, p.z); uv.push(sp * 4, (t + 1) * 2);
        }
      }
      for (let i = 0; i < nS; i++) for (let j = 0; j < nT; j++) {
        const q = i * (nT + 1) + j, r = q + nT + 1;
        idx.push(q, r, q + 1, q + 1, r, r + 1);
      }
      const g = new BufferGeometry();
      g.setAttribute('position', new Float32BufferAttribute(pos, 3));
      g.setAttribute('uv', new Float32BufferAttribute(uv, 2));
      g.setIndex(idx); g.computeVertexNormals();
      grp.add(new Mesh(g, mat));
    }
    return grp;
  }

  // short guard-hair strands scattered over the body: soft silhouette + coat texture
  function hairStrands(pos, idx, nor, col, flow, hlen, density, seed, kind, both = false) {
    const rnd = rng(seed);
    const P = PAL[kind], tip = P.tipLight, tabby = kind === 'tabby';
    const out = [], nrm = [], cc = [];
    const lift = 0.42, cl = Math.cos(lift), sl = Math.sin(lift);
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
      const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const area = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
      const exp = area * density;
      let k = Math.floor(exp);
      if (rnd() < exp - k) k++;
      for (let q = 0; q < k; q++) {
        let r1 = rnd(), r2 = rnd();
        if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
        const w0 = 1 - r1 - r2;
        const ia = a / 3, ib = b / 3, ic = c / 3;
        const L0 = hlen[ia] * w0 + hlen[ib] * r1 + hlen[ic] * r2;
        if (L0 < 0.0009) continue;
        const L = L0 * (0.65 + 0.7 * rnd());
        const px = pos[a] * w0 + pos[b] * r1 + pos[c] * r2, py = pos[a + 1] * w0 + pos[b + 1] * r1 + pos[c + 1] * r2, pz = pos[a + 2] * w0 + pos[b + 2] * r1 + pos[c + 2] * r2;
        let nx = nor[a] * w0 + nor[b] * r1 + nor[c] * r2, ny = nor[a + 1] * w0 + nor[b + 1] * r1 + nor[c + 1] * r2, nz = nor[a + 2] * w0 + nor[b + 2] * r1 + nor[c + 2] * r2;
        const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
        let fx = flow[a] * w0 + flow[b] * r1 + flow[c] * r2 + (rnd() - 0.5) * 0.5, fy = flow[a + 1] * w0 + flow[b + 1] * r1 + flow[c + 1] * r2 + (rnd() - 0.5) * 0.5, fz = flow[a + 2] * w0 + flow[b + 2] * r1 + flow[c + 2] * r2 + (rnd() - 0.5) * 0.5;
        const fd = fx * nx + fy * ny + fz * nz; fx -= nx * fd; fy -= ny * fd; fz -= nz * fd;
        const fl = Math.hypot(fx, fy, fz);
        if (fl < 1e-6) continue;
        fx /= fl; fy /= fl; fz /= fl;
        const dx = fx * cl + nx * sl, dy = fy * cl + ny * sl, dz = fz * cl + nz * sl;
        // half width
        const hw = 0.00011 + rnd() * 0.00006;
        const sx = (ny * dz - nz * dy) * hw, sy = (nz * dx - nx * dz) * hw, sz = (nx * dy - ny * dx) * hw;
        const droop = L * 0.22; // tips bend back toward the skin
        // winding chosen so the face normal points out of the skin (front-facing from outside)
        out.push(px + sx - nx * 0.0002, py + sy - ny * 0.0002, pz + sz - nz * 0.0002,
          px - sx - nx * 0.0002, py - sy - ny * 0.0002, pz - sz - nz * 0.0002,
          px + dx * L - nx * droop, py + dy * L - ny * droop, pz + dz * L - nz * droop);
        for (let r = 0; r < 3; r++) nrm.push(nx, ny, nz);
        const cr = col[a] * w0 + col[b] * r1 + col[c] * r2, cg = col[a + 1] * w0 + col[b + 1] * r1 + col[c + 1] * r2, cb = col[a + 2] * w0 + col[b + 2] * r1 + col[c + 2] * r2;
        const lum = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb;
        const v = 0.85 + rnd() * 0.3;
        cc.push(cr * 0.74 * v, cg * 0.74 * v, cb * 0.74 * v, cr * 0.74 * v, cg * 0.74 * v, cb * 0.74 * v);
        // agouti / sheen tips: lighter on tabby ground, brownish on black
        const tm = tabby ? (lum > 0.05 && lum < 0.35 ? 0.10 : 0.0) : 0.40;
        const tv = tabby ? 0.92 * v : v;
        cc.push(lerp(cr, tip[0], tm) * tv, lerp(cg, tip[1], tm) * tv, lerp(cb, tip[2], tm) * tv);
        if (both) { // same strand, opposite winding
          const o = out.length;
          out.push(out[o - 6], out[o - 5], out[o - 4], out[o - 9], out[o - 8], out[o - 7], out[o - 3], out[o - 2], out[o - 1]);
          for (let r = 0; r < 3; r++) nrm.push(nx, ny, nz);
          const q = cc.length;
          cc.push(cc[q - 6], cc[q - 5], cc[q - 4], cc[q - 9], cc[q - 8], cc[q - 7], cc[q - 3], cc[q - 2], cc[q - 1]);
        }
      }
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(out, 3));
    g.setAttribute('normal', new Float32BufferAttribute(nrm, 3));
    g.setAttribute('color', new Float32BufferAttribute(cc, 3));
    return g;
  }

  /* ------------------------------------------------------------ materials */
  const matCache = new Map();
  function furMaterial(kind) {
    if (matCache.has(kind)) return matCache.get(kind);
    const black = kind !== 'tabby';
    const m = new THREE.MeshPhysicalNodeMaterial({
      roughness: black ? 0.64 : 0.82, metalness: 0,
      sheen: black ? 0.85 : 0.6, sheenRoughness: black ? 0.48 : 0.55,
      sheenColor: new Color(black ? '#5c5045' : '#efe3cc'),
      specularIntensity: black ? 0.34 : 0.26,
    });
    const { positionGeometry: Pg, normalGeometry: Ng, vertexColor, texture, vec2, vec3, mx_noise_float, bumpMap, mix, float } = TSL;
    const vc = vertexColor().rgb;
    const fine = mx_noise_float(Pg.mul(vec3(1500.0, 1500.0, 1500.0))).mul(0.5).add(0.5);
    let detail = fine;
    if (U.plain) { m.colorNode = vc; matCache.set(kind, m); return m; }
    if (U.furTexture) {
      const tex = U.furTexture;
      const w = Ng.abs().pow(vec3(4.0, 4.0, 4.0));
      const wn = w.div(w.x.add(w.y).add(w.z));
      const s = 7.0;
      const a = texture(tex, vec2(Pg.z.mul(s), Pg.y.mul(s))).rgb, b = texture(tex, vec2(Pg.x.mul(s), Pg.z.mul(s))).rgb, c = texture(tex, vec2(Pg.x.mul(s), Pg.y.mul(s))).rgb;
      const lum = a.mul(wn.x).add(b.mul(wn.y)).add(c.mul(wn.z)).dot(vec3(0.2126, 0.7152, 0.0722)).div(0.012);
      detail = mix(fine, lum.clamp(0.0, 2.5).mul(0.5), float(0.7));
    }
    m.colorNode = vc.mul(detail.mul(black ? 0.55 : 0.42).add(black ? 0.72 : 0.79));
    m.normalNode = bumpMap(detail, float(black ? 0.00011 : 0.00008));
    matCache.set(kind, m);
    return m;
  }
  function skinMaterial(kind) {
    const key = 'skin-' + kind;
    if (matCache.has(key)) return matCache.get(key);
    const m = new THREE.MeshPhysicalNodeMaterial({ roughness: kind === 'tabby' ? 0.55 : 0.32, metalness: 0, clearcoat: 0.15, clearcoatRoughness: 0.4 });
    m.colorNode = TSL.vertexColor().rgb;
    matCache.set(key, m);
    return m;
  }
  function strandMaterial(kind) {
    const key = 'strand-' + kind;
    if (matCache.has(key)) return matCache.get(key);
    // same shading model as the coat (sheen darkens the base layer, so strands must share it or they glow)
    const m = furMaterial(kind).clone();
    m.normalNode = null;
    matCache.set(key, m);
    return m;
  }
  function earMaterial(kind) {
    const key = 'ear-' + kind;
    if (matCache.has(key)) return matCache.get(key);
    const m = furMaterial(kind).clone();
    m.side = THREE.DoubleSide;
    matCache.set(key, m);
    return m;
  }

  /* ---------------------------------------------------------------- build */
  const PRESETS = {
    tabby: { pose: 'loaf', coat: 'tabby', mirror: 1, chub: 1, h: 0.0021, look: [0.95, 0.60, 1.06],
      head: { scale: 1.10, cheek: 1.08, jowl: 1.06, muzzle: 1.0, eyeW: 0.0086, eyeH: 0.0067, earW: 0.039, earH: 0.040, lidDrop: 0 } },
    blackGreen: { pose: 'recline', coat: 'black', mirror: -1, chub: 1.04, h: 0.0021, collar: '#62b59c', look: [-0.46, 0.59, 1.04],
      head: { scale: 1.02, cheek: 1.05, jowl: 1.05, muzzle: 0.98, eyeW: 0.0082, eyeH: 0.0060, earW: 0.042, earH: 0.043, lidDrop: 0.0007 } },
    blackOrange: { pose: 'sprawl', coat: 'black', mirror: 1, chub: 1.2, h: 0.0022, collar: '#e3843a', look: [0.82, 0.71, 1.16], petal: { scale: 1.3, droop: 0.30 },
      head: { scale: 1.05, cheek: 1.13, jowl: 1.16, muzzle: 0.97, eyeW: 0.0085, eyeH: 0.0069, earW: 0.042, earH: 0.041, lidDrop: 0, earTilt: 0.45 } },
  };

  function build(name, opts = {}) {
    const cfg = Object.assign({}, PRESETS[name], opts);
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const outerStats = {};
    const A = assemble(cfg);
    const kind = cfg.coat;
    // bounds
    const bmin = [1e9, 1e9, 1e9], bmax = [-1e9, -1e9, -1e9];
    for (const p of A.prims) for (let i = 0; i < 3; i++) { bmin[i] = Math.min(bmin[i], p.box[i]); bmax[i] = Math.max(bmax[i], p.box[i + 3]); }
    for (let i = 0; i < 3; i++) { bmin[i] -= 0.012; bmax[i] += 0.012; }
    bmin[1] = Math.max(bmin[1], A.ground.y - 0.006);
    const h = cfg.h;
    const mesh = polygonize(A.f, bmin, bmax, h);
    const nor = refine(A.f, mesh.pos, h, mesh);
    const paint = painter(A, kind);
    const n = mesh.pos.length / 3, col = new Float32Array(n * 3), flow = new Float32Array(n * 3), hlen = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const c = paint(mesh.pos[i * 3], mesh.pos[i * 3 + 1], mesh.pos[i * 3 + 2]);
      col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
      flow[i * 3] = OUT.fx; flow[i * 3 + 1] = OUT.fy; flow[i * 3 + 2] = OUT.fz; hlen[i] = OUT.len;
    }
    // cavity / ambient occlusion from the distance field (creases between legs, chin, toes, under the belly)
    {
      const P = mesh.pos, steps = [0.004, 0.009, 0.016, 0.026];
      for (let i = 0; i < n; i++) {
        const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2], nx = nor[i * 3], ny = nor[i * 3 + 1], nz = nor[i * 3 + 2];
        const lf = mesh.locals[mesh.vblock[i]] || A.f;
        let occ = 0, wsum = 0;
        for (let k = 0; k < steps.length; k++) {
          const d = steps[k], f = (k < 1 ? lf : A.f)(x + nx * d, y + ny * d, z + nz * d);
          const wgt = 1 / (k + 1);
          occ += wgt * clamp((d - f) / d, 0, 1); wsum += wgt;
        }
        occ /= wsum;
        // the ground also occludes
        const gy = y - A.ground.y, gocc = ny < 0.2 ? (1 - sstep(0.0, 0.03, gy)) * 0.35 : 0;
        const ao = clamp(1 - occ * 0.95 - gocc, 0.32, 1);
        col[i * 3] *= ao; col[i * 3 + 1] *= ao; col[i * 3 + 2] *= ao;
      }
    }
    const geo = new BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(mesh.pos, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(new THREE.BufferAttribute(mesh.idx, 1));
    geo.computeBoundingSphere();
    const inner = new Group();
    const add = (g, m, name, shadow = true) => { const o = new Mesh(g, m); o.name = name; o.castShadow = shadow; o.receiveShadow = true; o.userData.keepUnbatched = true; inner.add(o); return o; };
    add(geo, furMaterial(kind), 'photo-matched-body');
    if (cfg.strands !== 0) {
      const sg = hairStrands(mesh.pos, mesh.idx, nor, col, flow, hlen, cfg.strands || (kind === 'tabby' ? 110000 : 150000), kind === 'tabby' ? 5 : 9, kind, kind !== 'tabby');
      add(sg, strandMaterial(kind), 'guard-hair', false);
      outerStats.strands = sg.attributes.position.count / (kind !== 'tabby' ? 6 : 3);
    }

    // ears
    for (const e of A.hs.ears) {
      const S = A.S;
      const base = V(e.base).applyMatrix4(A.HM);
      const dirIn = V([0, 0.004, -0.010]).applyQuaternion(A.hq).normalize();
      const q = A.hq.clone().multiply(new Quaternion().setFromEuler(new Euler(e.lean, e.turn, -e.tilt, 'ZYX')));
      const g = earGeometry({ W: e.W * S, H: e.H * S, D: 0.0070 * S, T: 0.0030 * S, curl: 0.0040 * S }, kind);
      const o = add(g, earMaterial(kind), 'ear');
      o.position.copy(base).addScaledVector(dirIn, 0.002);
      o.quaternion.copy(q);
    }
    // eyes
    const irisMat = new THREE.MeshPhysicalNodeMaterial({ map: irisTexture(kind), roughness: 0.18, clearcoat: 1, clearcoatRoughness: 0.04, specularIntensity: 0.8 });
    for (const e of A.eyes) {
      const g = eyeGeometry(e.R);
      const o = add(g, irisMat, 'eye', false);
      const ctr = e.surf.clone().addScaledVector(e.axis, -e.R * 1.13);
      let q = e.q.clone();
      if (cfg.look) {
        const want = V(cfg.look).sub(ctr).normalize();
        const ang = Math.acos(clamp(want.dot(e.axis), -1, 1)), lim = 0.30;
        const dir = ang > lim ? e.axis.clone().lerp(want, lim / ang).normalize() : want;
        q = new Quaternion().setFromUnitVectors(e.axis, dir).multiply(e.q);
      }
      o.quaternion.copy(q);
      o.position.copy(ctr);
    }
    // nose leather
    {
      const S = A.S;
      const tip = V([0, A.hs.nose.y, 0.07]).applyMatrix4(A.HM);
      const fwd = new Vector3(0, -0.10, 1).normalize().applyQuaternion(A.hq);
      const hit = rayHit(A.f, tip.toArray(), fwd.clone().negate().toArray(), 0.06, 0.0003);
      if (hit) {
        const gN = gradient(A.f, hit[0], hit[1], hit[2], 0.0008);
        const nrm = new Vector3(gN[0], gN[1], gN[2]).lerp(fwd, 0.4).normalize();
        const upH = new Vector3(0, 1, 0).applyQuaternion(A.hq);
        const xA = new Vector3().crossVectors(upH, nrm).normalize(), yA = new Vector3().crossVectors(nrm, xA);
        const o = add(noseGeometry(kind, A.hs.nose.w), skinMaterial(kind), 'nose-leather', false);
        o.quaternion.setFromRotationMatrix(new Matrix4().makeBasis(xA, yA, nrm));
        o.scale.setScalar(S);
        o.position.set(...hit).addScaledVector(nrm, -0.0011 * S);
      }
    }
    // mouth + philtrum, whiskers
    {
      const S = A.S;
      const onFace = (hx, hy, lift = 0.00025) => {
        const p = V([hx, hy, 0.08]).applyMatrix4(A.HM);
        const d = new Vector3(0, 0, -1).applyQuaternion(A.hq);
        const hit = rayHit(A.f, p.toArray(), d.toArray(), 0.08, 0.0003);
        if (!hit) return null;
        const g = gradient(A.f, hit[0], hit[1], hit[2], 0.0006);
        return [hit[0] + g[0] * lift, hit[1] + g[1] * lift, hit[2] + g[2] * lift];
      };
      const lineMat = new THREE.MeshStandardNodeMaterial({ color: kind === 'tabby' ? '#3b2f2a' : '#050505', roughness: 0.7 });
      const ny = A.hs.nose.y - 0.0040;
      const paths = [[[0, ny], [0, -0.0205]], [[0, -0.0205], [0.0040, -0.0228], [0.0085, -0.0238], [0.0125, -0.0228]], [[0, -0.0205], [-0.0040, -0.0228], [-0.0085, -0.0238], [-0.0125, -0.0228]]];
      if (kind === 'tabby') for (const path of paths) {
        const pts = path.map(([x, y]) => onFace(x, y)).filter(Boolean);
        if (pts.length >= 2) add(strandGeometry(pts, 0.00045 * S, 0.00038 * S, 5, 14), lineMat, 'mouth-line', false);
      }
      const rnd = rng(kind === 'tabby' ? 11 : 23);
      const wPale = new THREE.MeshStandardNodeMaterial({ color: '#f1eee6', roughness: 0.45 });
      const wDark = new THREE.MeshStandardNodeMaterial({ color: '#2a2725', roughness: 0.45 });
      const wGeos = { pale: [], dark: [] };
      for (const s of [1, -1]) {
        for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) {
          const hx = s * (0.0072 + k * 0.0026), hy = -0.0148 - r * 0.0028;
          const root = onFace(hx, hy, -0.0004);
          if (!root) continue;
          const out = new Vector3(s * 1, 0.10 - r * 0.16, 0.42 - k * 0.20 + (rnd() - 0.5) * 0.12).normalize().applyQuaternion(A.hq);
          const down = new Vector3(0, -1, 0).applyQuaternion(A.hq);
          const len = (0.058 + rnd() * 0.018) * S;
          const P0 = V(root), P1 = P0.clone().addScaledVector(out, len * 0.4), P2 = P0.clone().addScaledVector(out, len * 0.75).addScaledVector(down, len * 0.06), P3 = P0.clone().addScaledVector(out, len).addScaledVector(down, len * 0.16);
          const pale = kind === 'tabby' || (k === 0 && r === 1);
          wGeos[pale ? 'pale' : 'dark'].push(strandGeometry([P0, P1, P2, P3].map((v) => v.toArray()), 0.00034 * S, 0.00009 * S, 4, 16));
        }
        // brow whiskers
        for (let k = 0; k < 3; k++) {
          const root = onFace(s * (0.0100 + k * 0.0030), 0.0165 + k * 0.0006, -0.0003);
          if (!root) continue;
          const out = new Vector3(s * (0.45 + k * 0.2), 0.75, 0.55).normalize().applyQuaternion(A.hq);
          const len = (0.026 + k * 0.006) * S;
          const P0 = V(root), P2 = P0.clone().addScaledVector(out, len);
          wGeos[kind === 'tabby' ? 'pale' : 'dark'].push(strandGeometry([P0.toArray(), P0.clone().lerp(P2, 0.5).toArray(), P2.toArray()], 0.00026 * S, 0.00007 * S, 4, 10));
        }
      }
      for (const key of ['pale', 'dark']) if (wGeos[key].length) {
        const g = U.mergeGeometries ? U.mergeGeometries(wGeos[key], false) : wGeos[key][0];
        add(g, key === 'pale' ? wPale : wDark, 'whiskers', false);
      }
    }
    if (cfg.collar) {
      const cg = collarGroup(A, cfg.collar, cfg.pose === 'sprawl' ? 7 : 3, cfg.petal);
      cg.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.userData.keepUnbatched = true; } });
      cg.name = 'petal-scrunchie-collar';
      inner.add(cg);
    }
    inner.rotation.y = Math.PI; // face -z like the previous cats
    const outer = new Group();
    outer.add(inner);
    outer.userData.buildMs = Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0);
    outer.userData.vertices = n;
    outer.userData.triangles = mesh.idx.length / 3;
    outer.userData.strands = outerStats.strands || 0;
    return outer;
  }

  return { build, PRESETS, POSES, _internal: { assemble, polygonize, refine, painter } };
}
