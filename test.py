import fssa
import numpy as np

def mock_scaling_f(x):
    """Mock scaling function"""
    return np.exp(-(x + 1.0)**2)

def mock_scaled_data(l, rho, rho_c=0.5, nu=2.5, zeta=1.5):
    """Generate scaled data from mock scaling function"""
    return np.transpose(
        np.power(l, zeta / nu) *
        mock_scaling_f(
            np.outer(
                (rho - rho_c), np.power(l, 1 / nu)
            )
        )
    )

rhos = np.linspace(-0.5, 0.8, num=200)
ls = np.logspace(1, 3, num=5).astype(np.int64)
a = mock_scaled_data(ls, rhos)

da = a * 0.1

ret = fssa.autoscale(ls, rhos, a, da, rho_c0=0.5, nu0=2.5, zeta0=1.5, fix_rho_c=True)
print(ret)
print(ret.rho, ret.drho)
print(ret.nu, ret.dnu)
print(ret.zeta, ret.dzeta)
print(ret.fun)
