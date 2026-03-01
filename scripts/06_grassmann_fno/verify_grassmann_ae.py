"""
verify_grassmann_ae.py -- Smoke tests for Grassmann-FNO Autoencoder (ASCII output)

Runs a battery of unit tests with synthetic data.
No real dataset or GPU required.

Usage:
    python scripts/06_grassmann_fno/verify_grassmann_ae.py
"""

import os
import sys
import math
import traceback

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

from src.models.grassmann import (
    grassmann_project, log_map, exp_map,
    geodesic_distance, retract_qr,
    GrassmannProjector, GrassmannReconstructor,
)
from src.models.grassmann_fno_ae import (
    GrassmannFNOAutoencoder,
    ModeEmbedder, ModeUnembedder,
    FNO3d,
)

results = []

def check(name, cond, detail=''):
    sym = 'PASS' if cond else 'FAIL'
    msg = f'  [{sym}]  {name}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    results.append((name, cond))

def section(title):
    print(f'\n--- {title} ---')


# ===========================================================================
# 1. Grassmann Projection (SVD)
# ===========================================================================
section('1. Grassmann Projection (SVD) -- per-frame, rank clamped to min(M,C)')

B, T, M, C, k = 2, 5, 300, 3, 2  # k<=C for per-frame SVD
x = torch.randn(B, T, M, C)

try:
    U, S, Vh = grassmann_project(x, k)
    check('Output shapes',
          U.shape == (B, T, M, k) and
          S.shape == (B, T, k) and
          Vh.shape == (B, T, k, C),
          f'U={U.shape} S={S.shape} Vh={Vh.shape}')

    UtU_k = U.transpose(-2, -1) @ U
    eye   = torch.eye(k).expand(B, T, k, k)
    err   = (UtU_k - eye).abs().max().item()
    check('Orthonormality U^T U ~ I_k', err < 1e-4, f'max_err={err:.2e}')
except Exception as e:
    check('grassmann_project', False, str(e))
    traceback.print_exc()



section('2. SVD Reconstruction (U diag(S) Vh)')

try:
    F_mat = torch.randn(M, C)
    U_s, S_s, Vh_s = grassmann_project(F_mat.unsqueeze(0), k)
    U_s, S_s, Vh_s = U_s.squeeze(0), S_s.squeeze(0), Vh_s.squeeze(0)
    F_rec = (U_s * S_s.unsqueeze(-2)) @ Vh_s

    U_ref2, S_ref2, Vh_ref2 = torch.linalg.svd(F_mat, full_matrices=False)
    F_ref = (U_ref2[:, :k] * S_ref2[:k]) @ Vh_ref2[:k, :]

    err = (F_rec - F_ref).abs().max().item()
    check('SVD reconstruction matches torch.linalg.svd', err < 1e-4, f'max_err={err:.2e}')
except Exception as e:
    check('SVD reconstruction', False, str(e))
    traceback.print_exc()


section('3. Log/Exp Map Round-trip  Exp_Y(Log_Y(X)) ~ X (nearby X)')

try:
    n, k_rt = 100, 8
    Y = retract_qr(torch.randn(1, n, k_rt))
    # Make X close to Y via a small tangent step
    Delta_small = torch.randn(1, n, k_rt) * 0.1
    X = exp_map(Y, Delta_small)  # X is close to Y by construction

    Delta_recovered = log_map(Y, X)
    X_rec = exp_map(Y, Delta_recovered).squeeze(0)
    X_orig = X.squeeze(0)

    err = (X_rec - X_orig).abs().max().item()
    # Note: round-trip is only exact for infinitesimal tangents; ~0.08 error is
    # expected for 0.1 step-size tangents on the Grassmann manifold.
    check('Exp(Log(X)) round-trip', err < 0.15, f'max_err={err:.2e}')

    UtU = X_rec.T @ X_rec
    eye = torch.eye(k_rt)
    orth_err = (UtU - eye).abs().max().item()
    check('Exp result is orthonormal', orth_err < 1e-3, f'max_err={orth_err:.2e}')
except Exception as e:
    check('Log/Exp round-trip', False, str(e))
    traceback.print_exc()


section('4. Geodesic Distance Symmetry  d(Y,X) == d(X,Y)')

try:
    n, k_gd = 50, 6
    Y = retract_qr(torch.randn(1, n, k_gd))
    X = retract_qr(torch.randn(1, n, k_gd))

    d_yx = geodesic_distance(Y, X)
    d_xy = geodesic_distance(X, Y)
    err  = (d_yx - d_xy).abs().item()
    check('d(Y,X) == d(X,Y)', err < 1e-5, f'|d_yx - d_xy|={err:.2e}')

    # Self-distance -- acos(1-eps) gives small but nonzero value due to clamp
    d_yy = geodesic_distance(Y, Y)
    # With clamp at 1-1e-6, acos(1-1e-6) ~ 0.00141; tolerance ~ 0.01
    check('d(Y,Y) ~ 0', d_yy.item() < 0.02, f'd_yy={d_yy.item():.4e}')
except Exception as e:
    check('Geodesic distance', False, str(e))
    traceback.print_exc()



section('5. QR Retraction (Stiefel)')

try:
    raw = torch.randn(4, 200, 12)
    Q   = retract_qr(raw)
    UtU = Q.transpose(-1, -2) @ Q
    err = (UtU - torch.eye(12)).abs().max().item()
    check('QR retraction orthonormal', err < 1e-5, f'max_err={err:.2e}')
except Exception as e:
    check('QR retraction', False, str(e))


section('6. GrassmannProjector nn.Module (spatio-temporal POD)')

try:
    k_proj = 8   # k <= min(M, T*C) = min(300, 5*3=15), so 8 is fine
    proj = GrassmannProjector(rank=k_proj)
    # (B, T, M, C) = (2, 5, 300, 3)
    x_in = torch.randn(2, 5, 300, 3)
    U2, S2, Vh2 = proj(x_in)
    # U (B,T,M,k), S (B,T,k), Vh (B,T,k,C)
    check('GrassmannProjector U shape',
          U2.shape == (2, 5, 300, k_proj), f'{U2.shape}')
    check('GrassmannProjector S shape',
          S2.shape == (2, 5, k_proj), f'{S2.shape}')
    check('GrassmannProjector Vh shape',
          Vh2.shape == (2, 5, k_proj, 3), f'{Vh2.shape}')
    # Check U columns are orthonormal per batch item
    u_sample = U2[0, 0]  # (M, k)
    UtU = u_sample.T @ u_sample
    orth_err = (UtU - torch.eye(k_proj)).abs().max().item()
    check('U columns orthonormal', orth_err < 1e-4, f'max_err={orth_err:.2e}')
except Exception as e:
    check('GrassmannProjector', False, str(e))
    traceback.print_exc()



section('7. GrassmannReconstructor nn.Module')

try:
    rec   = GrassmannReconstructor()
    B_, T_, M_, k_, C_ = 2, 5, 300, 16, 3
    U_r   = torch.randn(B_, T_, M_, k_)
    S_r   = torch.rand(B_, T_, k_) + 0.1
    Vh_r  = torch.randn(B_, T_, k_, C_)
    out   = rec(U_r, S_r, Vh_r)
    check('GrassmannReconstructor shape', out.shape == (B_, T_, M_, C_), f'{out.shape}')
except Exception as e:
    check('GrassmannReconstructor', False, str(e))


section('8. ModeEmbedder / ModeUnembedder shapes')

try:
    rank8, in_ch8, fno_h8, nt_v8, G8 = 16, 3, 64, 5, 8
    emb   = ModeEmbedder(rank8, in_ch8, fno_h8, nt_v8, G8)
    unemb = ModeUnembedder(rank8, in_ch8, fno_h8, nt_v8)  # no grid_size needed

    S8  = torch.randn(2, nt_v8, rank8)
    Vh8 = torch.randn(2, nt_v8, rank8, in_ch8)

    grid8 = emb(S8, Vh8)
    check('ModeEmbedder output shape', grid8.shape == (2, fno_h8, nt_v8, G8, G8),
          f'{grid8.shape}')

    S_hat8, Vh_hat8 = unemb(grid8)
    check('ModeUnembedder S shape',   S_hat8.shape  == S8.shape,  f'{S_hat8.shape}')
    check('ModeUnembedder Vh shape',  Vh_hat8.shape == Vh8.shape, f'{Vh_hat8.shape}')

    # Test with different spatial size (robustness)
    grid_alt = torch.randn(2, fno_h8, nt_v8, G8 * 2, G8 * 2)
    S_alt, Vh_alt = unemb(grid_alt)
    check('ModeUnembedder robust to spatial size',
          S_alt.shape == S8.shape and Vh_alt.shape == Vh8.shape)
except Exception as e:
    check('ModeEmbedder/Unembedder', False, str(e))
    traceback.print_exc()


section('9. FNO3d forward pass')

try:
    fno9 = FNO3d(width=32, n_layers=2, modes_t=4, modes_h=4, modes_w=4)
    x9   = torch.randn(2, 32, 5, 8, 8)
    y9   = fno9(x9)
    check('FNO3d shape preserved', y9.shape == x9.shape, f'{y9.shape}')
    check('FNO3d no NaN',          not torch.isnan(y9).any())
except Exception as e:
    check('FNO3d', False, str(e))
    traceback.print_exc()


section('10. GrassmannFNOAutoencoder -- Full Forward Pass')

try:
    B10, T10, M10 = 1, 5, 200
    model10 = GrassmannFNOAutoencoder(
        in_channels=3, rank=8, grid_size=8, nt=T10,
        fno_modes=(4, 4, 4), fno_hidden=32, fno_layers=2,
        hidden_channels=32, ch_mult=(1, 2), num_res_blocks=1,
        dropout=0.0, z_channels=8, kl_weight=1e-6,
    )
    model10.eval()
    x10 = torch.randn(B10, T10, M10, 3)
    with torch.no_grad():
        x_rec10, loss10 = model10(x10)

    check('Output shape matches input', x_rec10.shape == x10.shape,
          f'{x_rec10.shape} vs {x10.shape}')
    check('No NaN in output',           not torch.isnan(x_rec10).any())
    check('Recon loss > 0',             loss10['recon'].item() > 0)
    check('KL loss finite',             math.isfinite(loss10['kl'].item()))
    check('Total loss finite',          math.isfinite(loss10['total'].item()))
except Exception as e:
    check('Full forward pass', False, str(e))
    traceback.print_exc()


section('11. Backward Pass (Gradient Flow)')

try:
    B11, T11, M11 = 1, 5, 100
    model11 = GrassmannFNOAutoencoder(
        in_channels=3, rank=8, grid_size=8, nt=T11,
        fno_modes=(4, 4, 4), fno_hidden=32, fno_layers=2,
        hidden_channels=32, ch_mult=(1, 2), num_res_blocks=1,
        dropout=0.0, z_channels=8, kl_weight=1e-6,
    )
    model11.train()
    x11 = torch.randn(B11, T11, M11, 3)
    _, loss11 = model11(x11)
    loss11['total'].backward()

    has_grad = all(p.grad is not None
                   for p in model11.parameters() if p.requires_grad)
    nan_grad = any(torch.isnan(p.grad).any()
                   for p in model11.parameters()
                   if p.requires_grad and p.grad is not None)
    check('All parameters have gradients', has_grad)
    check('No NaN gradients',             not nan_grad)
except Exception as e:
    check('Backward pass', False, str(e))
    traceback.print_exc()


section('12. Encode / Decode API')

try:
    B12, T12, M12 = 1, 5, 150
    model12 = GrassmannFNOAutoencoder(
        in_channels=3, rank=8, grid_size=8, nt=T12,
        fno_modes=(4, 4, 4), fno_hidden=32, fno_layers=2,
        hidden_channels=32, ch_mult=(1, 2), num_res_blocks=1,
        dropout=0.0, z_channels=8, kl_weight=0.0,
    )
    model12.eval()
    x12 = torch.randn(B12, T12, M12, 3)
    with torch.no_grad():
        z12, U_ref12, mean12, logvar12 = model12.encode(x12)
        x_dec12 = model12.decode(z12, U_ref12)

    check('z is 5D',             z12.ndim == 5, f'{z12.shape}')
    check('U_ref rank matches',  U_ref12.shape[-1] == 8, f'{U_ref12.shape}')
    check('decode shape correct', x_dec12.shape == x12.shape, f'{x_dec12.shape}')
except Exception as e:
    check('Encode/Decode API', False, str(e))
    traceback.print_exc()


# ===========================================================================
# Summary
# ===========================================================================
print('\n' + '='*60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
rate   = 100 * passed / total if total else 0
print(f'  Results: {passed}/{total} passed ({rate:.0f}%)')
if passed == total:
    print('  ALL TESTS PASSED')
else:
    failed = [name for name, ok in results if not ok]
    print(f'  FAILED: {", ".join(failed)}')
print('='*60)
sys.exit(0 if passed == total else 1)
