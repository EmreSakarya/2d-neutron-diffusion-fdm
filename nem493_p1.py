import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

#malzeme verileri
D = 0.614014
Sa = 3.59817e-4
S = 1.0
# bölgenin boyutları cm 
Lx = 30.0
Ly = 24.0


def build_matrix(h, bc):
    # bc sırası: batı,doğu,güney, kuzey
    west, east, south, north = bc
    nx = round(Lx / h) + 1
    ny = round(Ly / h) + 1

    if west == 1:
        i_first = 0
    else:
        i_first = 1
    if east == 1:
        i_last = nx - 1
    else:
        i_last = nx - 2
    if south == 1:
        j_first = 0
    else:
        j_first = 1
    if north == 1:
        j_last = ny - 1
    else:
        j_last = ny - 2

    n_per_row = i_last - i_first + 1
    n_rows = j_last - j_first + 1
    n = n_per_row * n_rows

    a_center = 4 * D / h**2 + Sa
    a_neighbor = -D / h**2

    A = sp.lil_matrix((n, n))
    b = np.zeros(n)

    for j in range(j_first, j_last + 1):
        for i in range(i_first, i_last + 1):
            # node'un matristeki satır numarası
            k = (j - j_first) * n_per_row + (i - i_first)
            A[k, k] = a_center
            b[k] = S

            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for ii, jj in neighbors:
                
                if ii < 0:
                    ii = 1
                if ii > nx - 1:
                    ii = nx - 2
                if jj < 0:
                    jj = 1
                if jj > ny - 1:
                    jj = ny - 2

                
                
                if i_first <= ii <= i_last and j_first <= jj <= j_last:
                    kk = (jj - j_first) * n_per_row + (ii - i_first)
                    A[k, kk] = A[k, kk] + a_neighbor

    return A.tocsr(), b, nx, ny, i_first, i_last, j_first, j_last


def solve(h, bc):
    A, b, nx, ny, i_first, i_last, j_first, j_last = build_matrix(h, bc)
    u = spsolve(A.tocsc(), b)

    
    phi = np.zeros((ny, nx))
    k = 0
    for j in range(j_first, j_last + 1):
        for i in range(i_first, i_last + 1):
            phi[j, i] = u[k]
            k = k + 1

    x = np.linspace(-Lx / 2, Lx / 2, nx)
    y = np.linspace(-Ly / 2, Ly / 2, ny)
    # hata kontrol
    residual = np.max(np.abs(A @ u - b))
    return A, phi, x, y, residual


def exact(x, y, a, b, x0, y0, M=4001):
    phi = np.zeros((len(y), len(x)))
    n_values = np.arange(1, M + 1, 2)
    kn = n_values * np.pi / b
    sin_y = np.sin(np.outer(y - y0, kn))

    for m in range(1, M + 1, 2):
        km = m * np.pi / a
        coef = 16 * S / (np.pi**2 * m * n_values * (D * (km**2 + kn**2) + Sa))
        y_part = sin_y @ coef
        x_part = np.sin(km * (x - x0))
        phi = phi + np.outer(y_part, x_part)

    return phi


def exact_absorption(a, b, part, M=4001):
    total = 0.0
    n_values = np.arange(1, M + 1, 2)
    kn = n_values * np.pi / b
    for m in range(1, M + 1, 2):
        km = m * np.pi / a
        coef = 16 * S / (np.pi**2 * m * n_values * (D * (km**2 + kn**2) + Sa))
        integral = (2 / km) * (2 / kn)
        total = total + np.sum(coef * integral)
    return Sa * total * part


def side_leakage(p1, p2, h):
    total = 0.0
    m = len(p1)
    for k in range(m):
        current = D * (4 * p1[k] - p2[k]) / (2 * h)
        weight = h
        if k == 0 or k == m - 1:
            weight = h / 2
        total = total + current * weight
    return total


def balance(phi, h, bc):
    ny, nx = phi.shape

    absorption = 0.0
    for j in range(ny):
        for i in range(nx):
            weight = h * h
            if i == 0 or i == nx - 1:
                weight = weight / 2
            if j == 0 or j == ny - 1:
                weight = weight / 2
            absorption = absorption + Sa * phi[j, i] * weight

    # sızıntı sadece vakum kenarlarından
    leakage = [0.0, 0.0, 0.0, 0.0]
    if bc[0] == 0:
        leakage[0] = side_leakage(phi[:, 1], phi[:, 2], h)
    if bc[1] == 0:
        leakage[1] = side_leakage(phi[:, nx - 2], phi[:, nx - 3], h)
    if bc[2] == 0:
        leakage[2] = side_leakage(phi[1, :], phi[2, :], h)
    if bc[3] == 0:
        leakage[3] = side_leakage(phi[ny - 2, :], phi[ny - 3, :], h)

    return absorption, leakage


# case 1: dört kenar vakum, case 2: doğu ve güney yansıtıcı
cases = [(0, 0, 0, 0), (0, 1, 1, 0)]
boxes = [(30, 24, -15, -12), (60, 48, -15, -36)]
parts = [1.0, 0.25]

for c in range(2):
    bc = cases[c]
    a, b, x0, y0 = boxes[c]
    A, phi, x, y, residual = solve(1.0, bc)
    phi_exact = exact(x, y, a, b, x0, y0)
    absorption, leakage = balance(phi, 1.0, bc)

    phi_max = 0.0
    x_max = 0.0
    y_max = 0.0
    for j in range(len(y)):
        for i in range(len(x)):
            if phi[j, i] > phi_max:
                phi_max = phi[j, i]
                x_max = x[i]
                y_max = y[j]

    total_leakage = leakage[0] + leakage[1] + leakage[2] + leakage[3]
    source = S * Lx * Ly

    print("Case", c + 1)
    print("  unknowns =", A.shape[0], " nonzeros =", A.nnz)
    print("  residual max|A*phi - S| =", f"{residual:.2e}")
    print("  phi_max =", f"{phi_max:.4f}", "at (", x_max, ",", y_max, ")")
    print("  phi(0,0) =", f"{phi[12, 15]:.4f}", " phi(15,0) =", f"{phi[12, 30]:.4f}", " phi(0,-12) =", f"{phi[0, 15]:.4f}")
    print("  absorption =", f"{absorption:.3f}")
    print("  leakage W, E, S, N =", [round(float(v), 2) for v in leakage], " total =", f"{total_leakage:.3f}")
    print("  balance gap =", f"{100 * (source - absorption - total_leakage) / source:.2f} %")

    exact_abs = exact_absorption(a, b, parts[c])
    print("  exact absorption =", f"{exact_abs:.3f}", " exact leakage =", f"{source - exact_abs:.3f}")

    A_half, phi_half, x_half, y_half, residual_half = solve(0.5, bc)
    absorption_half, leakage_half = balance(phi_half, 0.5, bc)
    gap_half = source - absorption_half - leakage_half[0] - leakage_half[1] - leakage_half[2] - leakage_half[3]
    print("  balance gap at h = 0.5 cm =", f"{100 * gap_half / source:.2f} %")
    print("  exact peak =", f"{np.max(phi_exact):.4f}", " max|FD - exact| =", f"{np.max(np.abs(phi - phi_exact)):.4f}")

    print("  flux every 5 cm in x and 4 cm in y:")
    for yy in [8, 4, 0, -4, -8, -12]:
        row = []
        for xx in [-10, -5, 0, 5, 10, 15]:
            row.append(round(float(phi[yy + 12, xx + 15]), 2))
        print("   y =", yy, row)

    A6 = solve(6.0, bc)[0].toarray()
    alpha = D / 36.0
    print("  A / alpha at h = 6 cm:")
    for row in A6:
        line = ""
        for v in row:
            line = line + f"{v / alpha:7.3f}"
        print("  ", line)

hs = [3, 1.5, 1, 0.75, 0.5, 0.375, 0.25]
for c in range(2):
    bc = cases[c]
    a, b, x0, y0 = boxes[c]
    errors = []
    sizes = []
    for h in hs:
        A, phi, x, y, residual = solve(h, bc)
        phi_exact = exact(x, y, a, b, x0, y0)
        errors.append(np.max(np.abs(phi - phi_exact)))
        sizes.append(A.shape[0])

    print("Case", c + 1, "mesh refinement")
    for k in range(len(hs)):
        if k == 0:
            print("  h =", hs[k], " n =", sizes[k], " error =", f"{errors[k]:.3e}")
        else:
            p = np.log(errors[k - 1] / errors[k]) / np.log(hs[k - 1] / hs[k])
            print("  h =", hs[k], " n =", sizes[k], " error =", f"{errors[k]:.3e}", " order =", f"{p:.2f}")
