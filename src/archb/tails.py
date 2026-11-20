"""Sequence 10: parsimony on rare events. Do Born-rule circuits (QCBM, cQCBM, C-QSM) need fewer parameters than
classical models to get the tails right, and does it depend on the bit encoding?

Encodings of one coordinate (k bits, B = 2^k bins in value order, edge bins open):
  'u'  B equal bins of [-L, L] (L = 8 in 1D, 6 per coordinate in 2D), Gray labels (sequences 1-9)
  'g'  sign + log-magnitude: magnitude bins [m_j, m_{j+1}), m_0 = 0, m_j = a rho^(j-1) up to `top`, the last one open;
       label = sign bit (1 = positive, MSB) then the magnitude index j in plain binary. A law geometric in j,
       P(j) ∝ r^j, is then a product law over the bits (r^j = prod_i (r^(2^i))^(b_i)): power-law tails become
       (piecewise) product states, cheap for every bit-based model (circuit, RBM, MPS) and out of reach of light-tailed
       mixtures.
2D: coordinates concatenated (coordinate 0 most significant), joint bins row-major, as archb.multivar.

Truths (exact cell probabilities; Student-t cells by Gauss-Legendre on theta = arctan(y), where the integrand
f_t(tan theta) sec^2 theta is bounded and smooth, also in JAX for the models):
  level 1 (i.i.d.)  t3 (Student-t, nu = 3), jump (0.99 N(0, 1) + 0.01 N(-5, 1): a rare crash mode), gauss (N(0, 1.5^2):
                    light-tailed control); 2D: tcop (bivariate t, nu = 3, rho = 0.5: tail dependence), gcop (Gaussian
                    copula rho = 0.5 with t3 margins: same margins, no tail dependence)
  level 2 (Markov)  art (x' = 0.7 v + 0.5 t3), arch (x' = sqrt(0.3 + 0.7 v^2) N(0, 1): conditional tails, power-law
                    stationary tail); v = the value of the current bin
  level 3 (memory)  crash (calm N(0.3 v, 0.5^2) / crash N(-1 + 0.3 v, 2^2), P(calm -> crash) = 0.01, crash kept with
                    0.8: rare events in short clusters), msart (sequence 4 msar with t3 innovations)

Model families (kind strings; 'static' = level 1 laws, 'cond' = kernels Q[x, x'] for levels 2-3):
  qB0L<L>, qA2L<L>  circuits of archb.cqsm with memory m = 0, 0 or 2 ancillas, L layers (cross gates for d = 2); static:
                    the input-controlled gates are absent (the law of input 0), so only the other angles are parameters
  rbm<H>            RBM on the label bits (static: n + H + nH; cond: archb.multivar conditional RBM)
  mix<K>            Gaussian mixture (static 1D 3K - 1, 2D bivariate 6K - 1; cond 1D as archb.ckernels, 6K - 2)
  mixt<K>           Student-t mixture, nu_k = 1 + exp(h_k) (static 1D 4K - 1; cond 1D 7K - 2): the classical baseline
                    built for heavy tails
  mps<chi>, nmps<chi>  real Born MPS (p = psi^2 / Z) and nonnegative MPS (p = prod A / Z) over the label bits, bond chi
                    (static only): the classical control for the Born rule itself
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from jax.scipy.stats import norm as jnorm
from scipy.stats import norm
from scipy.stats import t as student

from archb import ckernels, cqsm
from archb import multivar as MV
from archb.targets import gray_labels

jax.config.update("jax_enable_x64", True)

BIG = 1e4
QT = 8  # Gauss-Legendre nodes per bin, Student-t cells
TRUTH_NODES = 32


# ------------------------------------------------------------------------------------------------ grids
def grid(enc, n=6, d=1):
    k = n // d
    B = 2**k
    h = B // 2
    if enc == "u":
        L = 8.0 if d == 1 else 6.0
        e = np.linspace(-L, L, B + 1)
        w = e[1] - e[0]
        lab1 = gray_labels(k, 1)
        vals = np.concatenate([[e[1] - w], 0.5 * (e[1:-2] + e[2:-1]), [e[-2] + w]])
    else:
        a, top = (0.1, 30.0) if d == 1 else (0.25, 12.0)
        rho = (top / a) ** (1 / (h - 2))
        m = np.concatenate([[0.0], a * rho ** np.arange(h - 1)])  # m_0 .. m_{h-1}; magnitude bin h-1 is open
        e = np.concatenate([-m[:0:-1], [0.0], m[1:]])
        e = np.concatenate([[-np.inf], e, [np.inf]])
        mid = np.concatenate([0.5 * (m[:-1] + m[1:]), [m[-1] * np.sqrt(rho)]])  # magnitude values j = 0 .. h-1
        vals = np.concatenate([-mid[::-1], mid])
        i = np.arange(B)
        lab1 = np.where(i >= h, (1 << (k - 1)) | (i - h), h - 1 - i)
    e = e.astype(float)
    e[0], e[-1] = -np.inf, np.inf
    C = MV.coords(d, k)
    lab = sum(lab1[C[:, c]] << (k * (d - 1 - c)) for c in range(d))
    bits = ((lab[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.float64)
    em = e.copy()
    em[0], em[-1] = -BIG, BIG
    return {"enc": enc, "n": n, "d": d, "k": k, "B": B, "K": B**d, "edges": e, "edges_m": em, "values": vals,
            "lab": lab, "bits": bits, "coords": C}


# ------------------------------------------------------------------------------------------------ Student-t cells
def t_cells(mu, sig, nu, e, nodes=QT):
    """P[..., j] of the location-scale Student-t (any common batch shape of mu, sig, nu) over the bins of edges e
    (B + 1, outer edges +-BIG or +-inf); quadrature in theta = arctan((x - mu) / sig), rows renormalized. JAX."""
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    mu, sig, nu = (jnp.asarray(x)[..., None] for x in (mu, sig, nu))
    th = jnp.arctan((jnp.asarray(e) - mu) / sig)  # (..., B + 1)
    a, b = th[..., :-1, None], th[..., 1:, None]
    x = 0.5 * (b - a) * xg + 0.5 * (b + a)  # (..., B, Q)
    y = jnp.tan(x)
    nu_ = nu[..., None]
    logc = gammaln((nu_ + 1) / 2) - gammaln(nu_ / 2) - 0.5 * jnp.log(nu_ * jnp.pi)
    g = jnp.exp(logc - (nu_ + 1) / 2 * jnp.log1p(y**2 / nu_)) * (1 + y**2)
    P = jnp.sum(0.5 * (b - a) * wg * g, axis=-1)
    P = jnp.maximum(P, 0.0)
    return P / jnp.sum(P, axis=-1, keepdims=True)


def norm_cells(mu, sig, e):
    mu, sig = jnp.asarray(mu)[..., None], jnp.asarray(sig)[..., None]
    P = jnp.maximum(jnp.diff(jnorm.cdf((jnp.asarray(e) - mu) / sig), axis=-1), 0.0)
    return P / jnp.sum(P, axis=-1, keepdims=True)


# ------------------------------------------------------------------------------------------------ truths
LEVEL1 = {"t3": 1, "jump": 1, "gauss": 1, "tcop": 2, "gcop": 2}  # target -> d
LEVEL2 = ("art", "arch")
LEVEL3 = ("crash", "msart")
ART, ARCH = {"c": 0.7, "s": 0.5, "nu": 3.0}, {"a": 0.3, "b": 0.7}
CRASH = {"mu": (0.0, -1.0), "c": (0.3, 0.3), "sig": (0.5, 2.0), "nu": (np.inf, np.inf), "go": 0.01, "stay": 0.8}
MSART = {"mu": (0.0, 0.0), "c": (0.95, 0.7), "sig": (0.3, 1.0), "nu": (3.0, 3.0), "stay": (0.98, 0.98)}


def _biv_t_cells(nu, rho, e, nodes=TRUTH_NODES, s1=1.0, s2=1.0):
    """Bivariate t (scales s1, s2, correlation rho) over the cells of edges e (both coordinates): numpy."""
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    th = np.arctan(e / s1)
    a, b = th[:-1, None], th[1:, None]
    x1 = np.tan(0.5 * (b - a) * xg + 0.5 * (b + a))  # (B, Q), coordinate 1 in units of s1
    w = 0.5 * (b - a) * wg * student.pdf(x1, nu) * (1 + x1**2)
    cs = np.sqrt((nu + x1**2) * (1 - rho**2) / (nu + 1))
    cdf = student.cdf((e[None, None, :] / s2 - rho * x1[..., None]) / cs[..., None], nu + 1)  # (B, Q, B + 1)
    P = np.einsum("jq,jqk->jk", w, np.diff(cdf, axis=-1))
    return P / P.sum()


def truth_static(target, G):
    """Exact law over the bins (K,) of a level 1 target."""
    e = G["edges"]
    if target == "t3":
        return np.diff(student.cdf(e, 3))
    if target == "jump":
        return 0.99 * np.diff(norm.cdf(e)) + 0.01 * np.diff(norm.cdf(e, -5.0, 1.0))
    if target == "gauss":
        return np.diff(norm.cdf(e, 0.0, 1.5))
    if target == "tcop":
        return _biv_t_cells(3.0, 0.5, e).ravel()
    H = np.clip(norm.ppf(student.cdf(e, 3)), -BIG, BIG)
    return np.asarray(MV.bvn_cells(0.0, 0.0, 1.0, 1.0, 0.5, H, H, 24)).ravel()


def _rows(mu, sig, nu, e):
    """Rows of a location-scale kernel: Student-t with nu (inf = normal); numpy / scipy, exact CDFs."""
    if np.isinf(nu):
        return np.diff(norm.cdf((e[None, :] - mu[:, None]) / sig[:, None]), axis=1)
    return np.diff(student.cdf((e[None, :] - mu[:, None]) / sig[:, None], nu), axis=1)


def truth_kernel(target, G):
    v = G["values"]
    if target == "art":
        P = _rows(ART["c"] * v, np.full_like(v, ART["s"]), ART["nu"], G["edges"])
    else:
        P = _rows(0.0 * v, np.sqrt(ARCH["a"] + ARCH["b"] * v**2), np.inf, G["edges"])
    return P / P.sum(1, keepdims=True)


def ms_T(mu, c, sig, nu, Q, G, xp=np):
    """Input-output HMM T[x, h, x', h'] = Q[h, h'] K_h'[x, x'] of a 2-regime switching AR with Student-t (or normal)
    innovations. xp = np (exact truths) or jnp (family fits, Student-t cells by quadrature)."""
    v = G["values"]
    if xp is np:
        K = np.stack([_rows(mu[s] + c[s] * v, np.full_like(v, sig[s]), nu[s], G["edges"]) for s in range(2)])
        K /= K.sum(2, keepdims=True)
        return np.einsum("hk,kxy->xhyk", Q, K)
    v = jnp.asarray(v)
    K = t_cells(mu[:, None] + c[:, None] * v[None, :], jnp.broadcast_to(sig[:, None], (2, len(v))),
                jnp.broadcast_to(nu[:, None], (2, len(v))), jnp.asarray(G["edges_m"]))
    return jnp.einsum("hk,kxy->xhyk", Q, K)


def truth_T(target, G):
    if target == "crash":
        p = CRASH
        Q = np.array([[1 - p["go"], p["go"]], [1 - p["stay"], p["stay"]]])
    else:
        p = MSART
        Q = np.array([[p["stay"][0], 1 - p["stay"][0]], [1 - p["stay"][1], p["stay"][1]]])
    return ms_T(np.array(p["mu"]), np.array(p["c"]), np.array(p["sig"]), np.array(p["nu"]), Q, G)


# ------------------------------------------------------------------------------------------------ model families
_KIND = re.compile(r"(qB0|qA2)L(\d+)|(rbm|mix|mixt|mps|nmps)(\d+)")


def parse(kind):
    m = _KIND.fullmatch(kind)
    if m is None:
        raise ValueError(kind)
    return (m.group(1), int(m.group(2))) if m.group(1) else (m.group(3), int(m.group(4)))


def _circ(fam, L, G):
    na = 0 if fam == "qB0" else 2
    gates = cqsm.gate_list(0, na, L, G["n"], G["d"], G["d"] > 1)
    return na, gates, ~cqsm.input_mask(0, na, L, G["n"], G["d"], G["d"] > 1)


def _mps_shapes(chi, n):
    dims = [1] + [chi] * (n - 1) + [1]
    return [(2, dims[i], dims[i + 1]) for i in range(n)]


def n_params(kind, G, cond):
    fam, s = parse(kind)
    n, d = G["n"], G["d"]
    if fam in ("qB0", "qA2"):
        na, gates, free = _circ(fam, s, G)
        return len(gates) if cond else int(free.sum())
    if fam == "rbm":
        return n + n * n + s + 2 * n * s if cond else n + s + n * s
    if fam == "mix":
        return 6 * s - 2 if cond else (3 * s - 1 if d == 1 else 6 * s - 1)
    if fam == "mixt":
        return 7 * s - 2 if cond else 4 * s - 1
    return sum(int(np.prod(sh)) for sh in _mps_shapes(s, n))


def _split(theta, sizes):
    out, o = [], 0
    for z in sizes:
        out.append(theta[o:o + z])
        o += z
    return out


def static_probs(theta, kind, G):
    """Law over the bins (K,), JAX."""
    fam, s = parse(kind)
    theta = jnp.asarray(theta)
    lab, n, d = jnp.asarray(G["lab"]), G["n"], G["d"]
    if fam in ("qB0", "qA2"):
        na, gates, free = _circ(fam, s, G)
        full = jnp.zeros(len(gates)).at[np.flatnonzero(free)].set(theta)
        K = cqsm.label_kraus(full, 0, na, s, n, d, d > 1)
        return jnp.sum(jnp.abs(K[0]) ** 2, axis=(1, 2, 3))[lab]
    if fam == "rbm":
        b, c, W = _split(theta, [n, s, n * s])
        U = jnp.asarray(G["bits"])
        return jax.nn.softmax(U @ b + jnp.sum(jax.nn.softplus(c[None, :] + U @ W.reshape(n, s)), axis=1))
    if fam in ("mps", "nmps"):
        shapes = _mps_shapes(s, n)
        A = [t.reshape(sh) for t, sh in zip(_split(theta, [int(np.prod(sh)) for sh in shapes]), shapes)]
        if fam == "nmps":
            A = [jax.nn.softplus(t) for t in A]
        psi = A[0][:, 0, :]  # (2, chi): amplitudes of the first bit
        for t in A[1:]:
            psi = jnp.einsum("pi,yij->pyj", psi, t).reshape(-1, t.shape[2])  # label order, MSB first
        psi = psi[:, 0]
        w = psi**2 if fam == "mps" else psi
        return (w / jnp.sum(w))[lab]
    e = jnp.asarray(G["edges_m"])
    if d == 2:  # bivariate Gaussian mixture
        a, c, f, r = _split(theta, [s - 1, 2 * s, 2 * s, s])
        pi = jax.nn.softmax(jnp.concatenate([jnp.zeros(1), a]))
        c, sg = c.reshape(s, 2), jnp.exp(f.reshape(s, 2))
        P = MV.bvn_cells(c[:, 0], c[:, 1], sg[:, 0], sg[:, 1], 0.99 * jnp.tanh(r), e, e)
        return jnp.einsum("k,kij->ij", pi, P).ravel()
    if fam == "mix":
        a, c, f = _split(theta, [s - 1, s, s])
        return jax.nn.softmax(jnp.concatenate([jnp.zeros(1), a])) @ norm_cells(c, jnp.exp(f), e)
    a, c, f, h = _split(theta, [s - 1, s, s, s])
    return jax.nn.softmax(jnp.concatenate([jnp.zeros(1), a])) @ t_cells(c, jnp.exp(f), 1 + jnp.exp(h), e)


def cond_probs(theta, kind, G):
    """Kernel Q[x, x'] (K, K) in bin order, JAX (1D for the mixtures)."""
    fam, s = parse(kind)
    theta = jnp.asarray(theta)
    lab, n = jnp.asarray(G["lab"]), G["n"]
    if fam in ("qB0", "qA2"):
        na = 0 if fam == "qB0" else 2
        K = cqsm.label_kraus(theta, 0, na, s, n, G["d"], G["d"] > 1)
        return jnp.sum(jnp.abs(K) ** 2, axis=(2, 3, 4))[lab][:, lab]
    if fam == "rbm":
        return MV._rbm(theta, s, jnp.asarray(G["bits"]))
    v, e = jnp.asarray(G["values"]), jnp.asarray(G["edges_m"])
    if fam == "mix":  # floored and renormalized: a row whose components all left the grid stays a law
        P = ckernels._mix(theta, s, v, e) + 1e-15
        return P / jnp.sum(P, axis=1, keepdims=True)
    a, b, c, dd, f, g, h = _split(theta, [s - 1, s - 1, s, s, s, s, s])
    pi = jax.nn.softmax(jnp.concatenate([jnp.zeros(1), a])[None, :] +
                        jnp.concatenate([jnp.zeros(1), b])[None, :] * v[:, None], axis=1)
    mu = c[None, :] + dd[None, :] * v[:, None]
    sg = jnp.exp(f[None, :] + g[None, :] * jnp.abs(v)[:, None])
    return jnp.einsum("xk,xky->xy", pi, t_cells(mu, sg, jnp.broadcast_to(1 + jnp.exp(h), mu.shape), e))


def init(kind, seed, G, b_train, cond, start="marginal"):
    """Starts near the i.i.d. law of the training bins (circuits: small angles, ancillas in |+>, the final R_y layer at
    the training bit frequencies; start 'copy' = archb.cqsm copy init for kernels), RBM near the product of the bit
    marginals, mixtures with equal weights at training quantiles, MPS near the uniform law."""
    fam, s = parse(kind)
    rng = np.random.default_rng(seed)
    n = G["n"]
    p1 = np.clip(G["bits"][np.asarray(b_train)].mean(0), 1e-3, 1 - 1e-3)
    if fam in ("qB0", "qA2"):
        na, gates, free = _circ(fam, s, G)
        if cond and start == "copy":
            return cqsm.init_theta(seed, 0, na, s, n, 0.2, G["d"], G["d"] > 1)
        th = 0.02 * rng.normal(size=len(gates))
        th[[j for j, g in enumerate(gates) if g[3] == "anc_plus"]] += np.pi / 2
        th[-n:] = 2 * np.arcsin(np.sqrt(p1))
        return th if cond else th[free]
    if fam == "rbm":
        if cond:
            return np.concatenate([np.log(p1 / (1 - p1)), 0.01 * rng.normal(size=n * n), np.zeros(s),
                                   0.1 * rng.normal(size=2 * n * s)])
        return np.concatenate([np.log(p1 / (1 - p1)), np.zeros(s), 0.1 * rng.normal(size=n * s)])
    if fam in ("mps", "nmps"):
        out = []
        for sh in _mps_shapes(s, n):
            base = np.broadcast_to(np.eye(sh[1], sh[2]), sh) * (1.0 if fam == "mps" else 0.5)
            out.append((base + 0.1 * rng.normal(size=sh)).ravel())
        return np.concatenate(out)
    V = G["values"] if G["d"] == 1 else MV.joint_values({"values": np.tile(G["values"], (2, 1)), "d": 2, "k": G["k"]})
    x = V[np.asarray(b_train)]
    qs = (np.arange(s) + 0.5) / s
    if G["d"] == 2:
        c = np.stack([np.quantile(x[:, j], rng.permutation(qs)) for j in range(2)], 1)
        f = np.log(np.maximum(0.5 * (np.quantile(x, 0.75, 0) - np.quantile(x, 0.25, 0)), 0.05))
        return np.concatenate([0.01 * rng.normal(size=s - 1), (c + 0.05 * rng.normal(size=c.shape)).ravel(),
                               (np.tile(f, (s, 1)) + 0.05 * rng.normal(size=(s, 2))).ravel(), np.zeros(s)])
    c = np.quantile(x, qs) + 0.05 * rng.normal(size=s)
    f = np.full(s, np.log(max(0.5 * (np.quantile(x, 0.75) - np.quantile(x, 0.25)), 0.05))) + 0.05 * rng.normal(size=s)
    h = np.full(s, np.log(4.0))
    w = 0.01 * rng.normal(size=s - 1)
    if not cond:
        return np.concatenate([w, c, f] + ([h] if fam == "mixt" else []))
    z = lambda m: 0.01 * rng.normal(size=m)
    if fam == "mix":
        return np.concatenate([w, z(s - 1), c, z(s), f, z(s)])
    return np.concatenate([w, z(s - 1), c, z(s), f, z(s), h])


# ------------------------------------------------------------------------------------------------ memory models
def switch_M(theta, S, kind, G):
    """Bit-linear switch over S conditional kernels (as archb.multivar.switch_M): M (K, K, S, S)."""
    n, nk = G["n"], n_params(kind, G, True)
    theta = jnp.asarray(theta)
    W = theta[:S * (n + 1) * S].reshape(S, n + 1, S)
    U = jnp.asarray(G["bits"])
    P = jax.nn.softmax(W[None, :, 0, :] + jnp.einsum("xq,sqt->xst", U, W[:, 1:, :]), axis=2)
    o = S * (n + 1) * S
    Q = jnp.stack([cond_probs(theta[o + j * nk:o + (j + 1) * nk], kind, G) for j in range(S)])
    return jnp.einsum("xst,txy->xyst", P, Q)


def init_switch(theta_k, S, seed, G, noise=0.1, stay=3.0):
    rng = np.random.default_rng(seed)
    W = 0.01 * rng.normal(size=(S, G["n"] + 1, S))
    W[:, 0, :] += stay * np.eye(S)
    return np.concatenate([W.ravel()] + [np.asarray(theta_k) + noise * rng.normal(size=len(theta_k)) for _ in range(S)])


def cqsm_M(theta, m, G, n_anc=2, L=3):
    """C-QSM transfer matrices with this grid's labels, and f."""
    lab = jnp.asarray(G["lab"])
    K = cqsm.label_kraus(jnp.asarray(theta), m, n_anc, L, G["n"], 1, False)[lab][:, lab]
    return cqsm.transfer_from_kraus(K), cqsm.f_vec(m)


def ms_unpack(z):
    """Switching-AR family (10 parameters): mu, c, log sd, log(nu - 1), logit stay per regime; regime 1 = larger sd."""
    return z[0:2], z[2:4], jnp.exp(z[4:6]), 1 + jnp.exp(z[6:8]), jax.nn.sigmoid(z[8:10])


def ms_family_T(z, G):
    mu, c, sig, nu, stay = ms_unpack(z)
    Q = jnp.stack([jnp.stack([stay[0], 1 - stay[0]]), jnp.stack([1 - stay[1], stay[1]])])
    return ms_T(mu, c, sig, nu, Q, G, xp=jnp)


# ------------------------------------------------------------------------------------------------ tail scores
def tail_mask(p, alpha, side="both"):
    """Bins in the alpha tails of the law p (bins entirely beyond the alpha / 1 - alpha quantile; at least the outer bin)."""
    lo = np.cumsum(p) <= alpha
    hi = np.cumsum(p[::-1])[::-1] <= alpha
    lo[0], hi[-1] = True, True
    return {"both": lo | hi, "lower": lo, "upper": hi}[side]


def censored_kl(p, q, A):
    """KL between the laws censored outside A (Diks, Panchenko and van Dijk 2011): sum_A p log p / q + p(A^c) log of the
    ratio of the A^c masses. 0 iff q = p on A and q(A^c) = p(A^c)."""
    q = np.maximum(q, 1e-300)
    pA, qA = p[A], q[A]
    pc, qc = 1 - pA.sum(), max(1 - qA.sum(), 1e-300)
    return float(np.sum(np.where(pA > 0, pA * np.log(np.maximum(pA, 1e-300) / qA), 0.0)) +
                 (pc * np.log(pc / qc) if pc > 0 else 0.0))


def kl(p, q):
    return float(np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)), 0.0)))
