import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

D, Sa, S = 0.614014, 3.59817e-4, 1.0
Lx, Ly = 30.0, 24.0
cases = {1: (0, 0, 0, 0), 2: (0, 1, 1, 0)}
titles = {1: "Case 1: all sides vacuum", 2: "Case 2: W, N vacuum / E, S reflective"}
boxes = {1: (30, 24, -15, -12), 2: (60, 48, -15, -36)}


def T(n, lo, hi):
    m = n - 2 + lo + hi
    t = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m), format="lil")
    t[0, 1] -= lo
    t[-1, -2] -= hi
    return t.tocsr()


def solve(h, bc):
    w, e, s, n = bc
    nx, ny = round(Lx / h) + 1, round(Ly / h) + 1
    Tx, Ty = T(nx, w, e), T(ny, s, n)
    mx, my = Tx.shape[0], Ty.shape[0]
    A = D / h**2 * (sp.kron(sp.eye(my), Tx) + sp.kron(Ty, sp.eye(mx))) + Sa * sp.eye(mx * my)
    phi = np.zeros((ny, nx))
    phi[1 - s:ny - 1 + n, 1 - w:nx - 1 + e] = spsolve(A.tocsc(), np.full(mx * my, S)).reshape(my, mx)
    return A, phi, np.linspace(-15, 15, nx), np.linspace(-12, 12, ny)


def exact(x, y, a, b, x0, y0, M=4001):
    m = np.arange(1, M + 1, 2)
    km, kn = m * np.pi / a, m * np.pi / b
    c = 16 * S / (np.pi**2 * np.outer(m, m) * (D * (km[:, None]**2 + kn**2) + Sa))
    return np.sin(np.outer(y - y0, kn)) @ c.T @ np.sin(np.outer(x - x0, km)).T


def error(h, c):
    A, phi, x, y = solve(h, cases[c])
    return abs(phi - exact(x, y, *boxes[c])).max()


def balance(phi, h, bc):
    sides = [phi[:, 1:3], phi[:, -2:-4:-1], phi[1:3].T, phi[-2:-4:-1].T]
    leak = [(1 - r) * trapezoid(D * (4 * p[:, 0] - p[:, 1]) / (2 * h), dx=h) for p, r in zip(sides, bc)]
    return Sa * trapezoid(trapezoid(phi, dx=h), dx=h), leak


res = {}
for c, bc in cases.items():
    A, phi, x, y = solve(1.0, bc)
    ex = exact(x, y, *boxes[c])
    absr, leak = balance(phi, 1.0, bc)
    j, i = np.unravel_index(phi.argmax(), phi.shape)
    res[c] = (phi, x, y)
    print(f"Case {c}: n = {A.shape[0]}, nnz = {A.nnz}, phi_max = {phi.max():.4f} at ({x[i]:g}, {y[j]:g})")
    print(f"  phi(0,0) = {phi[12, 15]:.4f}, phi(15,0) = {phi[12, 30]:.4f}, phi(0,-12) = {phi[0, 15]:.4f}")
    print(f"  absorption = {absr:.3f}, leakage W E S N = {np.round(leak, 2)}, total = {sum(leak):.3f}")
    print(f"  max|FD - exact| = {abs(phi - ex).max():.4f}, exact peak = {ex.max():.4f}")
    print(f"  residual max|A*phi - S| = {abs(A @ phi[phi > 0] - S).max():.2e}")
    print(np.round(solve(6.0, bc)[0].toarray() * 36 / D, 3))

hs = [3, 1.5, 1, 0.75, 0.5, 0.375, 0.25]
err = {c: [error(h, c) for h in hs] for c in cases}
for c in cases:
    print(f"Case {c} errors:", np.round(err[c], 6), "order:", np.round(np.log(np.divide(err[c][:-1], err[c][1:])) / np.log(np.divide(hs[:-1], hs[1:])), 2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for k, c in enumerate(cases):
    phi, x, y = res[c]
    cf = ax[k].contourf(x, y, phi, 24, cmap="Blues")
    lv = np.arange(10, 90, 10) if c == 1 else np.arange(40, 320, 40)
    ax[k].clabel(ax[k].contour(x, y, phi, lv, colors="w", linewidths=0.6), fmt="%.0f", fontsize=7.5)
    ax[k].set(aspect="equal", xlabel="x [cm]", ylabel="y [cm]", title=titles[c])
    fig.colorbar(cf, ax=ax[k], shrink=0.82, label=r"$\phi$ [n/cm$^2$s]")
fig.savefig("fig_flux_maps.png", dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
xf, yf = np.linspace(-15, 15, 301), np.linspace(-12, 12, 241)
for c in cases:
    phi, x, y = res[c]
    col = f"C{c - 1}"
    ax[0].plot(xf, exact(xf, np.array([0.0]), *boxes[c])[0], col, label=f"Case {c} analytic")
    ax[0].plot(x[::2], phi[12, ::2], "o", c=col, mfc="w", label=f"Case {c} FD")
    ax[1].plot(yf, exact(np.array([0.0]), yf, *boxes[c])[:, 0], col)
    ax[1].plot(y[::2], phi[::2, 15], "o", c=col, mfc="w")
ax[0].set(xlabel="x [cm] (y = 0)", ylabel=r"$\phi$ [n/cm$^2$s]")
ax[1].set(xlabel="y [cm] (x = 0)")
ax[0].legend(loc="upper left")
fig.savefig("fig_profiles.png", dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5.6, 4.2))
for c in cases:
    ax.loglog(hs, err[c], "s-" if c == 1 else "o:", mfc="w", label=f"Case {c}")
ax.loglog(hs, 0.1 * np.array(hs)**2, "k--", lw=0.8, label=r"$h^2$")
ax.set_xticks(hs, [f"{h:g}" for h in hs])
ax.minorticks_off()
ax.set(xlabel="h [cm]", ylabel=r"max $|\phi_{FD}-\phi_{exact}|$ [n/cm$^2$s]")
ax.legend()
fig.savefig("fig_convergence.png", dpi=200, bbox_inches="tight")
