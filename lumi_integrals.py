#%%
# https://lhcmaskdoc.web.cern.ch/ipynbs/luminosity_formula/luminosity_formula/
# a modified version of G. Sterbini lumi model in https://gitlab.cern.ch/sterbini/lhcmaskdoc/-/blob/master/docs/ipynbs/luminosity_formula/luminosity_formula.ipynb
import numpy as np
from scipy import integrate
from scipy.constants import c as clight
import numba
from scipy.interpolate import interp1d
import scipy


class Particle:
    _masses = {"proton": 938.27208816e6, "electron": 510998.95}
    clight = clight

    def __init__(
        self,
        momentum=None,
        kind="proton",
        mass=None,
        energy=None,
        gamma=None,
        betagamma=None,
        beta=None,
    ):
        self.kind = kind
        if kind is not None and kind in self._masses:
            self.mass = self._masses[kind]
        elif self.mass is not None:
            self.mass = mass
        else:
            raise ValueError("Mass should be defined")
        if sum(x is not None for x in (momentum, mass, energy, gamma, betagamma)) == 1:
            if momentum is not None:
                self.momentum = momentum
            elif energy is not None:
                self.energy = energy
            elif gamma is not None:
                self.gamma = gamma
            elif beta is not None:
                self.beta = beta
            elif betagamma is not None:
                self.betagamma = betagamma
            else:
                self.momentum = 0
        else:
            raise ValueError("Momentum or energy ill defined")

    @property
    def energy(self):
        return np.sqrt(self.momentum**2 + self.mass**2)

    @energy.setter
    def energy(self, energy):
        self.momentum = np.sqrt(energy**2 - self.mass**2)

    @property
    def gamma(self):
        return self.energy / self.mass

    @gamma.setter
    def gamma(self, gamma):
        self.energy = gamma * self.mass

    @property
    def betagamma(self):
        return self.momentum / self.mass

    @betagamma.setter
    def betagamma(self, betagamma):
        self.momentum = betagamma * self.mass

    @property
    def beta(self):
        return self.momentum / self.energy

    @beta.setter
    def beta(self, beta):
        self.gamma = 1 / np.sqrt(1 - beta**2)


# Numba-optimized functions
@numba.njit
def beta(z, beta0, alpha_z0):
    return beta0 - 2 * alpha_z0 * z + (1 + alpha_z0**2) / beta0 * z**2

@numba.njit
def dispersion(z, d0, dp0):
    return d0 + z * dp0

@numba.njit
def sigma(beta, epsilon0, betagamma):
    return np.sqrt(beta * epsilon0 / betagamma)

@numba.njit
def theta_x_1(delta_z, CC_V_x_1, energy_tot1, CC_f_x_1, CC_phase_x_1):
    return CC_V_x_1 / energy_tot1 * np.sin(CC_phase_x_1 + 2 * np.pi * CC_f_x_1 / clight * delta_z)

@numba.njit
def theta_y_1(delta_z, CC_V_y_1, energy_tot1, CC_f_y_1, CC_phase_y_1):
    return CC_V_y_1 / energy_tot1 * np.sin(CC_phase_y_1 + 2 * np.pi * CC_f_y_1 / clight * delta_z)


@numba.njit
def theta_x_2(delta_z, CC_V_x_2, energy_tot2, CC_f_x_2, CC_phase_x_2):
    return CC_V_x_2 / energy_tot2 * np.sin(CC_phase_x_2 + 2 * np.pi * CC_f_x_2 / clight * delta_z)

@numba.njit
def theta_y_2(delta_z, CC_V_y_2, energy_tot2, CC_f_y_2, CC_phase_y_2):
    return CC_V_y_2 / energy_tot2 * np.sin(CC_phase_y_2 + 2 * np.pi * CC_f_y_2 / clight * delta_z)


@numba.njit
def kernel_double_integral(t, z, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params

    mx1_val = mx1(z, t, params)
    mx2_val = mx2(z, t, params)
    my1_val = my1(z, t, params)
    my2_val = my2(z, t, params)

    #sx1_z = sigma(beta(z, beta_x1, alpha_x1), epsilon_x1, betagamma_1)
    #sy1_z = sigma(beta(z, beta_y1, alpha_y1), epsilon_y1, betagamma_1)
    #sx2_z = sigma(beta(z, beta_x2, alpha_x2), epsilon_x2, betagamma_2)
    #sy2_z = sigma(beta(z, beta_y2, alpha_y2), epsilon_y2, betagamma_2)
    sx1_z = sx1(z, params) #sigma(beta(z, beta_x1, alpha_x1), epsilon_x1, betagamma_1) + (dispersion(z, br_1 * dx_1, br_1 * dpx_1) * deltap_p0_1)
    sy1_z = sy1(z, params) #sigma(beta(z, beta_y1, alpha_y1), epsilon_y1, betagamma_1) + (dispersion(z, br_1 * dy_1, br_1 * dpy_1) * deltap_p0_1)
    sx2_z = sx2(z, params) #sigma(beta(z, beta_x2, alpha_x2), epsilon_x2, betagamma_2) + (dispersion(z, br_2 * dx_2, br_2 * dpx_2) * deltap_p0_2)
    sy2_z = sy2(z, params) #sigma(beta(z, beta_y2, alpha_y2), epsilon_y2, betagamma_2) + (dispersion(z, br_2 * dy_2, br_2 * dpy_2) * deltap_p0_2)

    exp_term = -((mx1_val - mx2_val) ** 2) / (sx1_z ** 2 + sx2_z ** 2) - (my1_val - my2_val) ** 2 / (sy1_z ** 2 + sy2_z ** 2) - (-br_1 * clight * t + z) ** 2 / (sigma_z1**2) - (br_2 * clight * t + z) ** 2 / (sigma_z2**2)
    return np.exp(0.5 * exp_term) / np.sqrt((sx1_z ** 2 + sx2_z ** 2) * (sy1_z ** 2 + sy2_z ** 2)) / sigma_z1 / sigma_z2


##@numba.njit
#def Cq2(q):
#    Gamma = scipy.special.gamma
#    if q<1:
#        return (2*np.sqrt(np.pi))/((3-q)*np.sqrt(1-q))*(Gamma((1)/(1-q)))/(Gamma((3-q)/(2*(1-q))))
#    elif q==1:
#        return np.sqrt(np.pi)
#    elif q<3:
#        return (np.sqrt(np.pi))/(np.sqrt(q-1))*(Gamma((3-q)/(2*(q-1))))/(Gamma((1)/(q-1)))
#    else:
#        raise Exception('Please q<3!!!')
    
@numba.njit
def Cq2(q, gamma_1, gamma_2):
    #if q < 1:
    return (2 * np.sqrt(np.pi)) / ((3 - q) * np.sqrt(1 - q)) * gamma_1 / gamma_2
    #elif q == 1:
    #    return np.sqrt(np.pi)
    #elif q < 3:
    #    return (np.sqrt(np.pi)) / (np.sqrt(q - 1)) * gamma_1 / gamma_2
    #else:
    #    raise Exception('Please ensure q < 3!!!')

@numba.njit
def eq2(x,q):
    #print(q)
    #if q==1:
    #    return np.exp(x)
    if (1+(1-q)*x)>0:
        return (1+(1-q)*x)**(1/(1-q))
    else:
        #return 0
        return np.zeros_like(np.exp(x))
    
@numba.njit
def eq2(x, q):
    result = (1 + (1 - q) * x)
    return np.where(result > 0, result ** (1 / (1 - q)), 0.0)
    
@numba.njit
def qGauss2(x, mu, sigma, q, gamma_1, gamma_2, A=1):
    S = sigma * 4.0 *np.sqrt(2.0)
    b = 10.0 / S**2
    #b = 1.0/(sigma**2 * (5-3*q))
    #x, mu, q, b, A
    return A*np.sqrt(b)/Cq2(q, gamma_1, gamma_2)*eq2(-b*(x-mu)**2,q)



@numba.njit
def rho(x, mux, sigmax):
    return 1/np.sqrt(2*np.pi)/sigmax * np.exp(-(x-mux)**2/2/sigmax**2)

@numba.njit
def kernel_double_integral_pdf(t, z, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params

    mx1_val = mx1(z, t, params)
    mx2_val = mx2(z, t, params)
    my1_val = my1(z, t, params)
    my2_val = my2(z, t, params)

    #sx1_z = sigma(beta(z, beta_x1, alpha_x1), epsilon_x1, betagamma_1)
    #sy1_z = sigma(beta(z, beta_y1, alpha_y1), epsilon_y1, betagamma_1)
    #sx2_z = sigma(beta(z, beta_x2, alpha_x2), epsilon_x2, betagamma_2)
    #sy2_z = sigma(beta(z, beta_y2, alpha_y2), epsilon_y2, betagamma_2)
    sx1_z = sx1(z, params) #sigma(beta(z, beta_x1, alpha_x1), epsilon_x1, betagamma_1) + (dispersion(z, br_1 * dx_1, br_1 * dpx_1) * deltap_p0_1)
    sy1_z = sy1(z, params) #sigma(beta(z, beta_y1, alpha_y1), epsilon_y1, betagamma_1) + (dispersion(z, br_1 * dy_1, br_1 * dpy_1) * deltap_p0_1)
    sx2_z = sx2(z, params) #sigma(beta(z, beta_x2, alpha_x2), epsilon_x2, betagamma_2) + (dispersion(z, br_2 * dx_2, br_2 * dpx_2) * deltap_p0_2)
    sy2_z = sy2(z, params) #sigma(beta(z, beta_y2, alpha_y2), epsilon_y2, betagamma_2) + (dispersion(z, br_2 * dy_2, br_2 * dpy_2) * deltap_p0_2)

    #exp_term = -((mx1_val - mx2_val) ** 2) / (sx1_z ** 2 + sx2_z ** 2) - (my1_val - my2_val) ** 2 / (sy1_z ** 2 + sy2_z ** 2) - (-br_1 * clight * t + z) ** 2 / (sigma_z1**2) - (br_2 * clight * t + z) ** 2 / (sigma_z2**2)
    #return np.exp(0.5 * exp_term) / np.sqrt((sx1_z ** 2 + sx2_z ** 2) * (sy1_z ** 2 + sy2_z ** 2)) / sigma_z1 / sigma_z2

    exp_term = -((mx1_val - mx2_val) ** 2) / (sx1_z ** 2 + sx2_z ** 2) - (my1_val - my2_val) ** 2 / (sy1_z ** 2 + sy2_z ** 2) #- (-br_1 * clight * t + z) ** 2 / (sigma_z1**2) - (br_2 * clight * t + z) ** 2 / (sigma_z2**2)
    x_y_factor = np.exp(0.5 * exp_term) / np.sqrt((sx1_z ** 2 + sx2_z ** 2) * (sy1_z ** 2 + sy2_z ** 2)) #/ sigma_z1 / sigma_z2
    #return x_y_factor*rho(z, br_1 * clight * t, sigma_z1)*rho(z,-br_2 * clight * t, sigma_z2)
    return x_y_factor*qGauss2(z, br_1 * clight * t, sigma_z1, q=q, gamma_1=gamma_1, gamma_2=gamma_2)*qGauss2(z,-br_2 * clight * t, sigma_z2, q=q, gamma_1=gamma_1, gamma_2=gamma_2)

@numba.njit
def kernel_single_integral(z, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params

    mx1_val = x_1 + px_1 * z
    mx2_val = x_2 + px_2 * z
    my1_val = y_1 + py_1 * z
    my2_val = y_2 + py_2 * z

    sx1_z = sx1(z, params)#sigma(beta(z, beta_x1, alpha_x1), epsilon_x1, betagamma_1) + (dispersion(z, br_1 * dx_1, br_1 * dpx_1) * deltap_p0_1)
    sy1_z = sy1(z, params)#sigma(beta(z, beta_y1, alpha_y1), epsilon_y1, betagamma_1) + (dispersion(z, br_1 * dy_1, br_1 * dpy_1) * deltap_p0_1)
    sx2_z = sx2(z, params)#sigma(beta(z, beta_x2, alpha_x2), epsilon_x2, betagamma_2) + (dispersion(z, br_2 * dx_2, br_2 * dpx_2) * deltap_p0_2)
    sy2_z = sy2(z, params)#sigma(beta(z, beta_y2, alpha_y2), epsilon_y2, betagamma_2) + (dispersion(z, br_2 * dy_2, br_2 * dpy_2) * deltap_p0_2)

    exp_term = -((mx1_val - mx2_val) ** 2) / (sx1_z ** 2 + sx2_z ** 2) - (my1_val - my2_val) ** 2 / (sy1_z ** 2 + sy2_z ** 2) - ((br_1 + br_2) * z) ** 2 / (sigma_z1**2 + sigma_z2**2)
    return np.exp(0.5 * exp_term) / np.sqrt((sx1_z ** 2 + sx2_z ** 2) * (sy1_z ** 2 + sy2_z ** 2) * (sigma_z1**2 + sigma_z2**2))

@numba.njit
def sx1(z, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose,q, gamma_1, gamma_2
   ) = params    
    """The sigma_x of B1, quadratic sum of betatronic and dispersive sigma"""
    return np.sqrt(
            sigma(beta(z, beta_x1, alpha_x1), epsilon_x1, betagamma_1) ** 2
            + (dispersion(z, br_1 * dx_1, br_1 * dpx_1) * deltap_p0_1) ** 2
        )
@numba.njit
def sy1(z, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose,q,  gamma_1, gamma_2
   ) = params
    """The sigma_y of B1, quadratic sum of betatronic and dispersive sigma"""
    return np.sqrt(
            sigma(beta(z, beta_y1, alpha_y1), epsilon_y1, betagamma_1) ** 2
            + (dispersion(z, br_1 * dy_1, br_1 * dpy_1) * deltap_p0_1) ** 2
        )
@numba.njit
def sx2(z, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params
    """The sigma_x of B2, quadratic sum of betatronic and dispersive sigma"""
    return np.sqrt(
            sigma(beta(z, beta_x2, alpha_x2), epsilon_x2, betagamma_2) ** 2
            + (dispersion(z, br_2 * dx_2, br_2 * dpx_2) * deltap_p0_2) ** 2
        )
@numba.njit
def sy2(z, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params
    """The sigma_y of B2, quadratic sum of betatronic and dispersive sigma"""
    return np.sqrt(
            sigma(beta(z, beta_y2, alpha_y2), epsilon_y2, betagamma_2) ** 2
            + (dispersion(z, br_2 * dy_2, br_2 * dpy_2) * deltap_p0_2) ** 2
        )
    
@numba.njit
def mx1(z, t, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose,q, gamma_1, gamma_2
   ) = params
    """The mu_x of B1 as straight line"""
    return (
                x_1
                + R12_1 * theta_x_1(z - clight * t, CC_V_x_1, energy_tot1, CC_f_x_1, CC_phase_x_1)
                + (px_1 + R22_1 * theta_x_1(z - clight * t, CC_V_x_1, energy_tot1, CC_f_x_1, CC_phase_x_1)) * z
    )

@numba.njit
def my1(z, t, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose,q, gamma_1, gamma_2
   ) = params
    """The mu_y of B1 as straight line"""
    return (
                y_1
                + R34_1 * theta_y_1(z - clight * t, CC_V_y_1, energy_tot1, CC_f_y_1, CC_phase_y_1)
                + (py_1 + R44_1 * theta_y_1(z - clight * t, CC_V_y_1, energy_tot1, CC_f_y_1, CC_phase_y_1)) * z
    )
@numba.njit
def mx2(z, t, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose,q, gamma_1, gamma_2
   ) = params
    """The mu_x of B2 as straight line"""
    return (
                x_2
                + R12_2 * theta_x_2(z + clight * t, CC_V_x_2, energy_tot2, CC_f_x_2, CC_phase_x_2)
                + (px_2 + R22_2 * theta_x_2(z + clight * t, CC_V_x_2, energy_tot2, CC_f_x_2, CC_phase_x_2)) * z
    )

@numba.njit
def my2(z, t, params):
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params
    """The mu_y of B2 as straight line"""
    return (
                y_2
                + R34_2 * theta_y_2(z + clight * t, CC_V_y_2, energy_tot2, CC_f_y_2, CC_phase_y_2)
                + (py_2 + R44_2 * theta_y_2(z + clight * t, CC_V_y_2, energy_tot2, CC_f_y_2, CC_phase_y_2)) * z
    )


def luminosity(f=11245.5, nb=2808, N1=1.15e11, N2=1.15e11, x_1=0, x_2=0, y_1=0, y_2=0, px_1=142.5e-6, px_2=-142.5e-6, py_1=0, py_2=0, energy_tot1=7e12, energy_tot2=7e12, deltap_p0_1=0, deltap_p0_2=0, epsilon_x1=3.75e-6, epsilon_x2=3.75e-6, epsilon_y1=3.75e-6, epsilon_y2=3.75e-6, sigma_z1=7.55e-2, sigma_z2=7.55e-2, beta_x1=0.55, beta_x2=0.55, beta_y1=0.55, beta_y2=0.55, alpha_x1=0, alpha_x2=0, alpha_y1=0, alpha_y2=0, dx_1=0, dx_2=0, dy_1=0, dy_2=0, dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0, CC_V_x_1=0, CC_f_x_1=0, CC_phase_x_1=0, CC_V_x_2=0, CC_f_x_2=0, CC_phase_x_2=0, CC_V_y_1=0, CC_f_y_1=0, CC_phase_y_1=0, CC_V_y_2=0, CC_f_y_2=0, CC_phase_y_2=0, R12_1=0, R22_1=0, R34_1=0, R44_1=0, R12_2=0, R22_2=0, R34_2=0, R44_2=0, verbose=False, sigma_integration=3, flag_qGauss_pdf=False, q=1):
    #print("geeeere",flag_qGauss_pdf)
    particle_1 = Particle(energy=energy_tot1)
    betagamma_1 = particle_1.betagamma
    br_1 = particle_1.beta

    particle_2 = Particle(energy=energy_tot2)
    betagamma_2 = particle_2.betagamma
    br_2 = particle_2.beta

    c = clight

    if verbose:
        print(f"B1 momentum:{particle_1.momentum/1e9}")
        print(f"B2 momentum:{particle_2.momentum/1e9}")
        print(f"B1 betagamma:{betagamma_1}")
        print(f"B2 betagamma:{betagamma_2}")
        print(f"B1 beta:{br_1}")
        print(f"B2 beta:{br_2}")

    # module of B1 speed
    v0_1 = br_1 * c
    # paraxial hypothesis
    vx_1 = v0_1 * px_1
    vy_1 = v0_1 * py_1
    vz_1 = v0_1 * np.sqrt(1 - px_1**2 - py_1**2)
    v_1 = np.array([vx_1, vy_1, vz_1])

    v0_2 = br_2 * c  # module of B2 speed
    # Assuming counter rotating B2 ('-' sign)
    vx_2 = -v0_2 * px_2
    vy_2 = -v0_2 * py_2
    # assuming px_2**2+py_2**2 < 1
    vz_2 = -v0_2 * np.sqrt(1 - px_2**2 - py_2**2)
    v_2 = np.array([vx_2, vy_2, vz_2])

    if verbose:
        print(f"B1 velocity vector [c]:{v_1/c}")
        print(f"B2 velocity vector [c]:{v_2/c}")

    diff_v = v_1 - v_2
    cross_v = np.cross(v_1, v_2)

    # normalized to get 1 for the ideal case
    # NB we assume px_1 and py_1 constant along the z-slices
    # NOT TRUE FOR CC! In any case the Moeller efficiency is almost equal to 1 in most cases...
    Moeller_efficiency = (
        np.sqrt(c**2 * np.dot(diff_v, diff_v) - np.dot(cross_v, cross_v)) / c**2 / 2
    )

    sigma_z = np.max([sigma_z1, sigma_z2])

    if flag_qGauss_pdf:
        if q < 1:
            gamma_1 = scipy.special.gamma(1 / (1 - q))
            gamma_2 = scipy.special.gamma((3 - q) / (2 * (1 - q)))
        elif q == 1:
            # For q=1, we don't need gamma values
            gamma_1 = gamma_2 = 1
        elif q< 3:
            gamma_1 = scipy.special.gamma((3 - q) / (2 * (q - 1)))
            gamma_2 = scipy.special.gamma(1 / (q - 1))
    else:
        gamma_1 = gamma_2 = 0


    params = (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   )
    
    if verbose:
        print(f"Sigma_x1 [um]: {sx1(0, params)*1e6}")
        print(f"Sigma_x2 [um]: {sx2(0, params)*1e6}")
        print(f"Sigma_y1 [um]: {sy1(0, params)*1e6}")
        print(f"Sigma_y2 [um]: {sy2(0, params)*1e6}")



    if not [CC_V_x_1, CC_V_y_1, CC_V_x_2, CC_V_y_2] == [0, 0, 0, 0] or flag_qGauss_pdf == True:
        if verbose:
            print("Crab cavities are present or q-Gaussian selected!!")
        if verbose:
            eps = 1e-12
            print(f"Hor. crabbing angle B1 [murad] {(mx1(0,eps, params)-mx1(0,0, params))/eps/c}")
            print(f"Hor. crabbing angle B2 [murad] {(mx2(0,eps, params)-mx2(0,0, params))/eps/c}")
            print(f"Ver. crabbing angle B1 [murad] {(my1(0,eps, params)-my1(0,0, params))/eps/c}")
            print(f"Ver. crabbing angle B2 [murad] {(my2(0,eps, params)-my2(0,0, params))/eps/c}")
            eps = 1e-6
            print(f"Hor. net cros angle B1 [murad] {(mx1(eps,0, params)-mx1(0,0, params))/eps}")
            print(f"Hor. net cros angle B2 [murad] {(mx2(eps,0, params)-mx2(0,0, params))/eps}")
            print(f"Ver. cros angle B1 [murad] {(my1(eps,0, params)-my1(0,0, params))/eps}")
            print(f"Ver. cros angle B2 [murad] {(my2(eps,0, params)-my2(0,0, params))/eps}")

        #integral = integrate.dblquad(
        #    lambda t, z: kernel_double_integral(t, z, params),
        #    -sigma_integration * sigma_z,
        #    sigma_integration * sigma_z,
        #    lambda z: -sigma_integration * sigma_z / clight,
        #    lambda z: sigma_integration * sigma_z / clight,
        #)

        if not flag_qGauss_pdf:
            integral = integrate.dblquad(
                lambda t, z: kernel_double_integral(t, z, params),
                -sigma_integration * sigma_z,
                sigma_integration * sigma_z,
                lambda z: -sigma_integration * sigma_z / clight,
                lambda z: sigma_integration * sigma_z / clight,
            )

            L0 = f * N1 * N2 * nb * c / 2 / np.pi ** (2) * integral[0]
        else:
             
             integral = integrate.dblquad(
                lambda t, z: kernel_double_integral_pdf(t, z, params),
                -sigma_integration * sigma_z,
                sigma_integration * sigma_z,
                lambda z: -sigma_integration * sigma_z / clight,
                lambda z: sigma_integration * sigma_z / clight,
            )

             L0 = f * N1 * N2 * nb * c / 1 / np.pi * integral[0]           
    else:
        integral = integrate.quad(
            lambda z: kernel_single_integral(z, params), 
            -20 * sigma_z, 20 * sigma_z
        )
        #integral = integrate.dblquad(
        #    lambda t, z: kernel_double_integral(t, z, params),
        #    -sigma_integration * sigma_z,
        #    sigma_integration * sigma_z,
        #    lambda z: -sigma_integration * sigma_z / clight,
        #    lambda z: sigma_integration * sigma_z / clight,
        #)
        L0 = f * N1 * N2 * nb / np.sqrt(2) / np.pi ** (3 / 2) * integral[0]
        #L0 = f * N1 * N2 * nb * c / 2 / np.pi ** (2) * integral[0]
    result = L0 * Moeller_efficiency
    if verbose:
        print(f"Moeller efficiency: {Moeller_efficiency}")
        print(f"Integral Relative Error: {integral[1]/integral[0]}")
        print(f"==> Luminosity [Hz/cm^2]: {result}")
    return result



def luminosity_z(myz, f=11245.5, nb=2808, N1=1.15e11, N2=1.15e11, x_1=0, x_2=0, y_1=0, y_2=0, px_1=142.5e-6, px_2=-142.5e-6, py_1=0, py_2=0, energy_tot1=7e12, energy_tot2=7e12, deltap_p0_1=0, deltap_p0_2=0, epsilon_x1=3.75e-6, epsilon_x2=3.75e-6, epsilon_y1=3.75e-6, epsilon_y2=3.75e-6, sigma_z1=7.55e-2, sigma_z2=7.55e-2, beta_x1=0.55, beta_x2=0.55, beta_y1=0.55, beta_y2=0.55, alpha_x1=0, alpha_x2=0, alpha_y1=0, alpha_y2=0, dx_1=0, dx_2=0, dy_1=0, dy_2=0, dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0, CC_V_x_1=0, CC_f_x_1=0, CC_phase_x_1=0, CC_V_x_2=0, CC_f_x_2=0, CC_phase_x_2=0, CC_V_y_1=0, CC_f_y_1=0, CC_phase_y_1=0, CC_V_y_2=0, CC_f_y_2=0, CC_phase_y_2=0, R12_1=0, R22_1=0, R34_1=0, R44_1=0, R12_2=0, R22_2=0, R34_2=0, R44_2=0, verbose=False, sigma_integration=3, flag_qGauss_pdf=False, q=1):

    particle_1 = Particle(energy=energy_tot1)
    betagamma_1 = particle_1.betagamma
    br_1 = particle_1.beta

    particle_2 = Particle(energy=energy_tot2)
    betagamma_2 = particle_2.betagamma
    br_2 = particle_2.beta

    c = clight

    if verbose:
        print(f"B1 momentum:{particle_1.momentum/1e9}")
        print(f"B2 momentum:{particle_2.momentum/1e9}")
        print(f"B1 betagamma:{betagamma_1}")
        print(f"B2 betagamma:{betagamma_2}")
        print(f"B1 beta:{br_1}")
        print(f"B2 beta:{br_2}")

    # module of B1 speed
    v0_1 = br_1 * c
    # paraxial hypothesis
    vx_1 = v0_1 * px_1
    vy_1 = v0_1 * py_1
    vz_1 = v0_1 * np.sqrt(1 - px_1**2 - py_1**2)
    v_1 = np.array([vx_1, vy_1, vz_1])

    v0_2 = br_2 * c  # module of B2 speed
    # Assuming counter rotating B2 ('-' sign)
    vx_2 = -v0_2 * px_2
    vy_2 = -v0_2 * py_2
    # assuming px_2**2+py_2**2 < 1
    vz_2 = -v0_2 * np.sqrt(1 - px_2**2 - py_2**2)
    v_2 = np.array([vx_2, vy_2, vz_2])

    if verbose:
        print(f"B1 velocity vector [c]:{v_1/c}")
        print(f"B2 velocity vector [c]:{v_2/c}")

    diff_v = v_1 - v_2
    cross_v = np.cross(v_1, v_2)

    # normalized to get 1 for the ideal case
    # NB we assume px_1 and py_1 constant along the z-slices
    # NOT TRUE FOR CC! In any case the Moeller efficiency is almost equal to 1 in most cases...
    Moeller_efficiency = (
        np.sqrt(c**2 * np.dot(diff_v, diff_v) - np.dot(cross_v, cross_v)) / c**2 / 2
    )

    sigma_z = np.max([sigma_z1, sigma_z2])

    if flag_qGauss_pdf:
        if q < 1:
            gamma_1 = scipy.special.gamma(1 / (1 - q))
            gamma_2 = scipy.special.gamma((3 - q) / (2 * (1 - q)))
        elif q == 1:
            # For q=1, we don't need gamma values
            gamma_1 = gamma_2 = 1
        elif q< 3:
            gamma_1 = scipy.special.gamma((3 - q) / (2 * (q - 1)))
            gamma_2 = scipy.special.gamma(1 / (q - 1))
    else:
        gamma_1 = gamma_2 = 0

    params = (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1, 
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2, 
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2, 
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2, 
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1, 
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2, 
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   )
    
    if verbose:
        print(f"Sigma_x1 [um]: {sx1(0, params)*1e6}")
        print(f"Sigma_x2 [um]: {sx2(0, params)*1e6}")
        print(f"Sigma_y1 [um]: {sy1(0, params)*1e6}")
        print(f"Sigma_y2 [um]: {sy2(0, params)*1e6}")

    if not [CC_V_x_1, CC_V_y_1, CC_V_x_2, CC_V_y_2] == [0, 0, 0, 0] or flag_qGauss_pdf == True:
        if verbose:
            print("Crab cavities are present!!")
        if verbose:
            eps = 1e-12
            print(f"Hor. crabbing angle B1 [murad] {(mx1(0,eps, params)-mx1(0,0, params))/eps/c}")
            print(f"Hor. crabbing angle B2 [murad] {(mx2(0,eps, params)-mx2(0,0, params))/eps/c}")
            print(f"Ver. crabbing angle B1 [murad] {(my1(0,eps, params)-my1(0,0, params))/eps/c}")
            print(f"Ver. crabbing angle B2 [murad] {(my2(0,eps, params)-my2(0,0, params))/eps/c}")
            eps = 1e-6
            print(f"Hor. net cros angle B1 [murad] {(mx1(eps,0, params)-mx1(0,0, params))/eps}")
            print(f"Hor. net cros angle B2 [murad] {(mx2(eps,0, params)-mx2(0,0, params))/eps}")
            print(f"Ver. cros angle B1 [murad] {(my1(eps,0, params)-my1(0,0, params))/eps}")
            print(f"Ver. cros angle B2 [murad] {(my2(eps,0, params)-my2(0,0, params))/eps}")

        #integral = integrate.dblquad(
        #    lambda t, z: kernel_double_integral(t, z, params),
        #    -sigma_integration * sigma_z,
        #    sigma_integration * sigma_z,
        #    lambda z: -sigma_integration * sigma_z / clight,
        #    lambda z: sigma_integration * sigma_z / clight,
        #)

        if not flag_qGauss_pdf:
            integral = integrate.quad(
                lambda t: kernel_double_integral(t, myz, params), 
                -sigma_integration * sigma_z / clight, sigma_integration * sigma_z / clight,
            )
            L0 = f * N1 * N2 * nb * c / 2 / np.pi ** (2) * integral[0]
        else:
            integral = integrate.quad(
                lambda t: kernel_double_integral_pdf(t, myz, params), 
                -sigma_integration * sigma_z / clight, sigma_integration * sigma_z / clight,
            )
            L0 = f * N1 * N2 * nb * c / 1. / np.pi  * integral[0]
            
        
    else:
        #integral = integrate.quad(
        #    lambda z: kernel_single_integral(z, params), 
        #    -20 * sigma_z, 20 * sigma_z
        #)
        myint = kernel_single_integral(myz, params)
        L0 = f * N1 * N2 * nb / np.sqrt(2) / np.pi ** (3 / 2) * myint
    result = L0 * Moeller_efficiency
    if verbose:
        print(f"Moeller efficiency: {Moeller_efficiency}")
        #print(f"Integral Relative Error: {integral[1]/integral[0]}")
        #print(f"==> Luminosity [Hz/cm^2]: {result}")
    return result

def cmp_FWHM(z, L):    
    # Find the index of the maximum luminosity value
    max_index = np.argmax(L)
    L_max = L[max_index]

    # Find the half-maximum value
    half_max = L_max / 2

    # Interpolate to find the z-values at half-maximum
    interp_func = interp1d(L[:max_index] - half_max, z[:max_index], kind='linear', fill_value="extrapolate")
    z_half_max_left = interp_func(0)

    interp_func = interp1d(L[max_index:] - half_max, z[max_index:], kind='linear', fill_value="extrapolate")
    z_half_max_right = interp_func(0)

    # Calculate FWHM
    FWHM = z_half_max_right - z_half_max_left

    # Estimate the standard deviation for a Gaussian distribution
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return FWHM, sigma, z_half_max_left, z_half_max_right

def longitudinal_luminous_region(sigma_z_list,
          f=11245.5, nb=2808, N1=1.15e11, N2=1.15e11, 
           x_1=0, x_2=0, y_1=0, y_2=0, 
           px_1=142.5e-6, px_2=-142.5e-6, py_1=0, py_2=0, 
           energy_tot1=7e12, energy_tot2=7e12, 
           deltap_p0_1=0, deltap_p0_2=0, 
           epsilon_x1=3.75e-6, epsilon_x2=3.75e-6, 
           epsilon_y1=3.75e-6, epsilon_y2=3.75e-6, 
           sigma_z1=7.55e-2, sigma_z2=7.55e-2, 
           beta_x1=0.55, beta_x2=0.55, 
           beta_y1=0.55, beta_y2=0.55, 
           alpha_x1=0, alpha_x2=0, 
           alpha_y1=0, alpha_y2=0, 
           dx_1=0, dx_2=0, dy_1=0, dy_2=0, 
           dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0, 
           CC_V_x_1=0, CC_f_x_1=0, CC_phase_x_1=0, 
           CC_V_x_2=0, CC_f_x_2=0, CC_phase_x_2=0, 
           CC_V_y_1=0, CC_f_y_1=0, CC_phase_y_1=0, 
           CC_V_y_2=0, CC_f_y_2=0, CC_phase_y_2=0, 
           R12_1=0, R22_1=0, R34_1=0, R44_1=0, R12_2=0, R22_2=0, R34_2=0, R44_2=0, 
           verbose=False, sigma_integration=3, flag_qGauss_pdf=False, q=1, sigmaBOff_m2= 8.1e-30):
    
    longitudinal_density = [luminosity_z(z, f=f, nb=nb, N1=N1, N2=N2, 
           x_1=x_1, x_2=x_2, y_1=y_1, y_2=y_2, 
           px_1=px_1, px_2=px_2, py_1=py_1, py_2=py_2, 
           energy_tot1=energy_tot1, energy_tot2=energy_tot2, 
           deltap_p0_1=deltap_p0_1, deltap_p0_2=deltap_p0_2, 
           epsilon_x1=epsilon_x1, epsilon_x2=epsilon_x2, 
           epsilon_y1=epsilon_y1, epsilon_y2=epsilon_y2, 
           sigma_z1=sigma_z1, sigma_z2=sigma_z2,
              beta_x1=beta_x1, beta_x2=beta_x2,
                beta_y1=beta_y1, beta_y2=beta_y2,
                alpha_x1=alpha_x1, alpha_x2=alpha_x2,
                alpha_y1=alpha_y1, alpha_y2=alpha_y2,
                dx_1=dx_1, dx_2=dx_2, dy_1=dy_1, dy_2=dy_2,
                dpx_1=dpx_1, dpx_2=dpx_2, dpy_1=dpy_1, dpy_2=dpy_2,
                CC_V_x_1=CC_V_x_1, CC_f_x_1=CC_f_x_1, CC_phase_x_1=CC_phase_x_1,
                CC_V_x_2=CC_V_x_2, CC_f_x_2=CC_f_x_2, CC_phase_x_2=CC_phase_x_2,
                CC_V_y_1=CC_V_y_1, CC_f_y_1=CC_f_y_1, CC_phase_y_1=CC_phase_y_1,
                CC_V_y_2=CC_V_y_2, CC_f_y_2=CC_f_y_2, CC_phase_y_2=CC_phase_y_2,
                R12_1=R12_1, R22_1=R22_1, R34_1=R34_1, R44_1=R44_1, R12_2=R12_2, R22_2=R22_2, R34_2=R34_2, R44_2=R44_2,
                verbose=verbose, sigma_integration=sigma_integration, flag_qGauss_pdf=flag_qGauss_pdf, q=q) for z in sigma_z_list]

    L = np.array(longitudinal_density)
    z= sigma_z_list

    total_luminosity = np.trapz(L, z)
    mean_z = np.trapz(z * L, z) / total_luminosity

    rms = np.sqrt(np.trapz(z**2 * L, z) / total_luminosity)

    variance = np.trapz((z - mean_z)**2 * L, z) / total_luminosity
    std_dev = np.sqrt(variance)

    pileup_density = sigmaBOff_m2/f/nb * L
    peak_pileup_density = np.max(sigmaBOff_m2/f/nb * L)

    return longitudinal_density, rms, std_dev, total_luminosity, peak_pileup_density, pileup_density, mean_z


########################################################################################################
########################################################################################################
# Transverse luminous region
#
# The longitudinal luminous region above is obtained by evaluating the fully
# analytic kernel_single_integral(z, params) (valid when there are no crab
# cavities) at a series of z values: x, y and t have already been integrated
# out analytically in that kernel, leaving only the z-dependence.
#
# To get the *transverse* luminous region (the density of collision points as
# a function of the transverse offset x, or y, from the nominal collision
# axis) we cannot reuse that kernel directly, because it no longer carries any
# x or y dependence. Instead we redo the same kind of analytic reduction but
# keep x (or y) explicit:
#   - the other transverse coordinate (y when direction="x", x when
#     direction="y") is still integrated out analytically (product of two
#     Gaussians -> Gaussian in the difference of the means, combined width
#     sqrt(sy1^2+sy2^2))
#   - without crab cavities mx1, mx2, my1, my2 do not depend on the collision
#     time t, so the time integral can also be carried out analytically
#     exactly like it is for kernel_single_integral (same br_1~br_2
#     approximation, already used elsewhere in this module)
#   - what remains is a single integral over z, done numerically with
#     scipy.integrate.quad, exactly as luminosity() does to get the total
#     luminosity from kernel_single_integral.
#
# When crab cavities (or the q-Gaussian longitudinal profile) are active, mx1,
# mx2 (or my1, my2) depend on t as well, so no analytic t-integration is
# possible: in that case a general (slower) numerical double integral over
# (t, z) is used instead, following the same structure as
# kernel_double_integral / kernel_double_integral_pdf.
#
# In both cases, integrating the returned density over the position array
# reproduces the total luminosity returned by luminosity()/luminosity_z().

def _transverse_density_analytic(z, myu, direction, params):
    """Fast path (no crab cavities, no q-Gaussian): density at transverse
    offset myu, as a function of the remaining integration variable z."""
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1,
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2,
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2,
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2,
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1,
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2,
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params

    sx1_z = sx1(z, params)
    sx2_z = sx2(z, params)
    sy1_z = sy1(z, params)
    sy2_z = sy2(z, params)

    # without crab cavities mx1/mx2/my1/my2 do not depend on t
    mx1_val = mx1(z, 0.0, params)
    mx2_val = mx2(z, 0.0, params)
    my1_val = my1(z, 0.0, params)
    my2_val = my2(z, 0.0, params)

    if direction == "x":
        u_factor = (np.exp(-0.5 * (myu - mx1_val) ** 2 / sx1_z ** 2)
                    * np.exp(-0.5 * (myu - mx2_val) ** 2 / sx2_z ** 2)
                    / (2 * np.pi * sx1_z * sx2_z))
        v_factor = (np.exp(-0.5 * (my1_val - my2_val) ** 2 / (sy1_z ** 2 + sy2_z ** 2))
                    / np.sqrt(2 * np.pi * (sy1_z ** 2 + sy2_z ** 2)))
    else:
        u_factor = (np.exp(-0.5 * (myu - my1_val) ** 2 / sy1_z ** 2)
                    * np.exp(-0.5 * (myu - my2_val) ** 2 / sy2_z ** 2)
                    / (2 * np.pi * sy1_z * sy2_z))
        v_factor = (np.exp(-0.5 * (mx1_val - mx2_val) ** 2 / (sx1_z ** 2 + sx2_z ** 2))
                    / np.sqrt(2 * np.pi * (sx1_z ** 2 + sx2_z ** 2)))

    # analytic result of the t integral (br_1~br_2 approximation, consistent
    # with the one already used to build kernel_single_integral)
    z_factor = (np.exp(-0.5 * ((br_1 + br_2) * z) ** 2 / (sigma_z1 ** 2 + sigma_z2 ** 2))
                / np.sqrt(2 * np.pi * (sigma_z1 ** 2 + sigma_z2 ** 2)))

    return u_factor * v_factor * z_factor


def _transverse_density_numeric(t, z, myu, direction, params, flag_qGauss_pdf=False):
    """General path (crab cavities and/or q-Gaussian): integrand of the
    (t, z) double integral at fixed transverse offset myu."""
    (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1,
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2,
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2,
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2,
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1,
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2,
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   ) = params

    sx1_z = sx1(z, params)
    sx2_z = sx2(z, params)
    sy1_z = sy1(z, params)
    sy2_z = sy2(z, params)

    mx1_val = mx1(z, t, params)
    mx2_val = mx2(z, t, params)
    my1_val = my1(z, t, params)
    my2_val = my2(z, t, params)

    if direction == "x":
        u_factor = (np.exp(-0.5 * (myu - mx1_val) ** 2 / sx1_z ** 2)
                    * np.exp(-0.5 * (myu - mx2_val) ** 2 / sx2_z ** 2)
                    / (2 * np.pi * sx1_z * sx2_z))
        v_factor = (np.exp(-0.5 * (my1_val - my2_val) ** 2 / (sy1_z ** 2 + sy2_z ** 2))
                    / np.sqrt(2 * np.pi * (sy1_z ** 2 + sy2_z ** 2)))
    else:
        u_factor = (np.exp(-0.5 * (myu - my1_val) ** 2 / sy1_z ** 2)
                    * np.exp(-0.5 * (myu - my2_val) ** 2 / sy2_z ** 2)
                    / (2 * np.pi * sy1_z * sy2_z))
        v_factor = (np.exp(-0.5 * (mx1_val - mx2_val) ** 2 / (sx1_z ** 2 + sx2_z ** 2))
                    / np.sqrt(2 * np.pi * (sx1_z ** 2 + sx2_z ** 2)))

    if not flag_qGauss_pdf:
        z_t_factor = (np.exp(-0.5 * (-br_1 * clight * t + z) ** 2 / sigma_z1 ** 2
                              - 0.5 * (br_2 * clight * t + z) ** 2 / sigma_z2 ** 2)
                      / (2 * np.pi * sigma_z1 * sigma_z2))
    else:
        z_t_factor = (qGauss2(z, br_1 * clight * t, sigma_z1, q=q, gamma_1=gamma_1, gamma_2=gamma_2)
                      * qGauss2(z, -br_2 * clight * t, sigma_z2, q=q, gamma_1=gamma_1, gamma_2=gamma_2))

    return u_factor * v_factor * z_t_factor


def luminosity_u(myu, direction="x",
          f=11245.5, nb=2808, N1=1.15e11, N2=1.15e11,
           x_1=0, x_2=0, y_1=0, y_2=0,
           px_1=142.5e-6, px_2=-142.5e-6, py_1=0, py_2=0,
           energy_tot1=7e12, energy_tot2=7e12,
           deltap_p0_1=0, deltap_p0_2=0,
           epsilon_x1=3.75e-6, epsilon_x2=3.75e-6,
           epsilon_y1=3.75e-6, epsilon_y2=3.75e-6,
           sigma_z1=7.55e-2, sigma_z2=7.55e-2,
           beta_x1=0.55, beta_x2=0.55,
           beta_y1=0.55, beta_y2=0.55,
           alpha_x1=0, alpha_x2=0,
           alpha_y1=0, alpha_y2=0,
           dx_1=0, dx_2=0, dy_1=0, dy_2=0,
           dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0,
           CC_V_x_1=0, CC_f_x_1=0, CC_phase_x_1=0,
           CC_V_x_2=0, CC_f_x_2=0, CC_phase_x_2=0,
           CC_V_y_1=0, CC_f_y_1=0, CC_phase_y_1=0,
           CC_V_y_2=0, CC_f_y_2=0, CC_phase_y_2=0,
           R12_1=0, R22_1=0, R34_1=0, R44_1=0, R12_2=0, R22_2=0, R34_2=0, R44_2=0,
           verbose=False, sigma_integration=3, flag_qGauss_pdf=False, q=1):
    """Luminosity density [Hz/cm^2/m] generated at transverse offset `myu`
    from the nominal collision axis (myu measured along x if
    direction=='x', along y if direction=='y'). This is the transverse
    analogue of luminosity_z: integrating luminosity_u(myu) over myu
    reproduces the total luminosity given by luminosity()."""

    assert direction in ("x", "y"), "direction must be 'x' or 'y'"

    particle_1 = Particle(energy=energy_tot1)
    betagamma_1 = particle_1.betagamma
    br_1 = particle_1.beta

    particle_2 = Particle(energy=energy_tot2)
    betagamma_2 = particle_2.betagamma
    br_2 = particle_2.beta

    c = clight

    v0_1 = br_1 * c
    vx_1 = v0_1 * px_1
    vy_1 = v0_1 * py_1
    vz_1 = v0_1 * np.sqrt(1 - px_1**2 - py_1**2)
    v_1 = np.array([vx_1, vy_1, vz_1])

    v0_2 = br_2 * c
    vx_2 = -v0_2 * px_2
    vy_2 = -v0_2 * py_2
    vz_2 = -v0_2 * np.sqrt(1 - px_2**2 - py_2**2)
    v_2 = np.array([vx_2, vy_2, vz_2])

    diff_v = v_1 - v_2
    cross_v = np.cross(v_1, v_2)
    Moeller_efficiency = (
        np.sqrt(c**2 * np.dot(diff_v, diff_v) - np.dot(cross_v, cross_v)) / c**2 / 2
    )

    sigma_z = np.max([sigma_z1, sigma_z2])

    if flag_qGauss_pdf:
        if q < 1:
            gamma_1 = scipy.special.gamma(1 / (1 - q))
            gamma_2 = scipy.special.gamma((3 - q) / (2 * (1 - q)))
        elif q == 1:
            gamma_1 = gamma_2 = 1
        elif q < 3:
            gamma_1 = scipy.special.gamma((3 - q) / (2 * (q - 1)))
            gamma_2 = scipy.special.gamma(1 / (q - 1))
    else:
        gamma_1 = gamma_2 = 0

    params = (beta_x1, alpha_x1, epsilon_x1, betagamma_1, dx_1, dpx_1, deltap_p0_1,
    beta_x2, alpha_x2, epsilon_x2, betagamma_2, dx_2, dpx_2, deltap_p0_2,
    beta_y1, alpha_y1, epsilon_y1, betagamma_1, dy_1, dpy_1, deltap_p0_1,
    beta_y2, alpha_y2, epsilon_y2, betagamma_2, dy_2, dpy_2, deltap_p0_2,
    sigma_z1, sigma_z2, x_1, y_1, px_1, py_1, x_2, y_2, px_2, py_2,
    R12_1, R22_1, R34_1, R44_1, R12_2, R22_2, R34_2, R44_2,
    CC_V_x_1, CC_f_x_1, CC_phase_x_1, CC_V_y_1, CC_f_y_1, CC_phase_y_1,
    CC_V_x_2, CC_f_x_2, CC_phase_x_2, CC_V_y_2, CC_f_y_2, CC_phase_y_2,
    energy_tot1, energy_tot2, br_1, br_2, clight, sigma_integration, verbose, q, gamma_1, gamma_2
   )

    has_cc = not [CC_V_x_1, CC_V_y_1, CC_V_x_2, CC_V_y_2] == [0, 0, 0, 0]

    if (not has_cc) and (not flag_qGauss_pdf):
        integral = integrate.quad(
            lambda z: _transverse_density_analytic(z, myu, direction, params),
            -sigma_integration * sigma_z, sigma_integration * sigma_z,
        )
        L0 = 2.0 * f * N1 * N2 * nb * integral[0]
    else:
        integral = integrate.dblquad(
            lambda t, z: _transverse_density_numeric(t, z, myu, direction, params, flag_qGauss_pdf),
            -sigma_integration * sigma_z,
            sigma_integration * sigma_z,
            lambda z: -sigma_integration * sigma_z / clight,
            lambda z: sigma_integration * sigma_z / clight,
        )
        L0 = 2.0 * f * N1 * N2 * nb * clight * integral[0]

    result = L0 * Moeller_efficiency
    if verbose:
        print(f"Moeller efficiency: {Moeller_efficiency}")
        print(f"==> Luminosity density at {direction}={myu} [Hz/cm^2/m]: {result}")
    return result


def luminosity_x(myx, **kwargs):
    """Luminosity density [Hz/cm^2/m] as a function of the horizontal offset
    myx from the nominal collision axis. See luminosity_u for details."""
    return luminosity_u(myx, direction="x", **kwargs)


def luminosity_y(myy, **kwargs):
    """Luminosity density [Hz/cm^2/m] as a function of the vertical offset
    myy from the nominal collision axis. See luminosity_u for details."""
    return luminosity_u(myy, direction="y", **kwargs)


def transverse_luminous_region(pos_list, direction="x",
          f=11245.5, nb=2808, N1=1.15e11, N2=1.15e11,
           x_1=0, x_2=0, y_1=0, y_2=0,
           px_1=142.5e-6, px_2=-142.5e-6, py_1=0, py_2=0,
           energy_tot1=7e12, energy_tot2=7e12,
           deltap_p0_1=0, deltap_p0_2=0,
           epsilon_x1=3.75e-6, epsilon_x2=3.75e-6,
           epsilon_y1=3.75e-6, epsilon_y2=3.75e-6,
           sigma_z1=7.55e-2, sigma_z2=7.55e-2,
           beta_x1=0.55, beta_x2=0.55,
           beta_y1=0.55, beta_y2=0.55,
           alpha_x1=0, alpha_x2=0,
           alpha_y1=0, alpha_y2=0,
           dx_1=0, dx_2=0, dy_1=0, dy_2=0,
           dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0,
           CC_V_x_1=0, CC_f_x_1=0, CC_phase_x_1=0,
           CC_V_x_2=0, CC_f_x_2=0, CC_phase_x_2=0,
           CC_V_y_1=0, CC_f_y_1=0, CC_phase_y_1=0,
           CC_V_y_2=0, CC_f_y_2=0, CC_phase_y_2=0,
           R12_1=0, R22_1=0, R34_1=0, R44_1=0, R12_2=0, R22_2=0, R34_2=0, R44_2=0,
           verbose=False, sigma_integration=3, flag_qGauss_pdf=False, q=1, sigmaBOff_m2=8.1e-30):
    """Transverse analogue of longitudinal_luminous_region: scans luminosity_u
    over `pos_list` (a list/array of transverse offsets, in meters, along x if
    direction=='x' or along y if direction=='y') and returns the same tuple
    structure (density, rms, std_dev, total_luminosity, peak_pileup_density,
    pileup_density, mean_pos)."""

    transverse_density = [luminosity_u(u, direction=direction, f=f, nb=nb, N1=N1, N2=N2,
           x_1=x_1, x_2=x_2, y_1=y_1, y_2=y_2,
           px_1=px_1, px_2=px_2, py_1=py_1, py_2=py_2,
           energy_tot1=energy_tot1, energy_tot2=energy_tot2,
           deltap_p0_1=deltap_p0_1, deltap_p0_2=deltap_p0_2,
           epsilon_x1=epsilon_x1, epsilon_x2=epsilon_x2,
           epsilon_y1=epsilon_y1, epsilon_y2=epsilon_y2,
           sigma_z1=sigma_z1, sigma_z2=sigma_z2,
              beta_x1=beta_x1, beta_x2=beta_x2,
                beta_y1=beta_y1, beta_y2=beta_y2,
                alpha_x1=alpha_x1, alpha_x2=alpha_x2,
                alpha_y1=alpha_y1, alpha_y2=alpha_y2,
                dx_1=dx_1, dx_2=dx_2, dy_1=dy_1, dy_2=dy_2,
                dpx_1=dpx_1, dpx_2=dpx_2, dpy_1=dpy_1, dpy_2=dpy_2,
                CC_V_x_1=CC_V_x_1, CC_f_x_1=CC_f_x_1, CC_phase_x_1=CC_phase_x_1,
                CC_V_x_2=CC_V_x_2, CC_f_x_2=CC_f_x_2, CC_phase_x_2=CC_phase_x_2,
                CC_V_y_1=CC_V_y_1, CC_f_y_1=CC_f_y_1, CC_phase_y_1=CC_phase_y_1,
                CC_V_y_2=CC_V_y_2, CC_f_y_2=CC_f_y_2, CC_phase_y_2=CC_phase_y_2,
                R12_1=R12_1, R22_1=R22_1, R34_1=R34_1, R44_1=R44_1, R12_2=R12_2, R22_2=R22_2, R34_2=R34_2, R44_2=R44_2,
                verbose=verbose, sigma_integration=sigma_integration, flag_qGauss_pdf=flag_qGauss_pdf, q=q) for u in pos_list]

    L = np.array(transverse_density)
    pos = np.asarray(pos_list)

    total_luminosity = np.trapz(L, pos)
    mean_pos = np.trapz(pos * L, pos) / total_luminosity

    rms = np.sqrt(np.trapz(pos**2 * L, pos) / total_luminosity)

    variance = np.trapz((pos - mean_pos)**2 * L, pos) / total_luminosity
    std_dev = np.sqrt(variance)

    pileup_density = sigmaBOff_m2/f/nb * L
    peak_pileup_density = np.max(sigmaBOff_m2/f/nb * L)

    return transverse_density, rms, std_dev, total_luminosity, peak_pileup_density, pileup_density, mean_pos



if __name__ == "__main__":
    print(luminosity(verbose=True, sigma_z1=0.09, sigma_z2=0.09))
    print(luminosity(verbose=True, flag_qGauss_pdf=True, q=3.0/5.0, sigma_z1=0.0761, sigma_z2=0.0761))

    print()
    print()
    print()
    
    factor = np.sqrt((4-2**(8/5))/(np.log2(2)))
    sigma_z_list = np.linspace(-0.2, 0.2, 200)
    #longitudinal_density, rms, std_dev, mean_z, total_luminosity, peak_pileup_density, pileup_density
    longitudinal_density, rms, std_dev,  total_luminosity, peak_pileup_density, pileup_density, mean_z = longitudinal_luminous_region(sigma_z_list,
            f=11245.5, nb=2452, N1=1.6e11, N2=1.6e11, 
            x_1=0, x_2=0, y_1=0, y_2=0, 
            px_1=0.00018999999999999999, px_2=-0.00018999999999999999, py_1=0, py_2=0, 
            energy_tot1=6.8e12, energy_tot2=6.8e12, 
            deltap_p0_1=0, deltap_p0_2=0, 
            epsilon_x1=2.72930272e-06, epsilon_x2=2.72930272e-06, 
            epsilon_y1= 2.72930272e-06, epsilon_y2=2.72930272e-06, 
            sigma_z1= 0.0761, sigma_z2=0.0761, 
            beta_x1=0.3, beta_x2=0.3, 
            beta_y1=0.3, beta_y2=0.3, 
            alpha_x1=0, alpha_x2=0, 
            alpha_y1=0, alpha_y2=0, 
            dx_1=0, dx_2=0, dy_1=0, dy_2=0, 
            dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0, 
            CC_V_x_1=-3.4e6, CC_f_x_1=400e6, CC_phase_x_1=0, 
            CC_V_x_2=+3.4e6, CC_f_x_2=400e6, CC_phase_x_2=0, 
            CC_V_y_1=-3.4e6, CC_f_y_1=400e6, CC_phase_y_1=0, 
            CC_V_y_2=+3.4e6, CC_f_y_2=400e6, CC_phase_y_2=0, 
            R12_1=46.66, R22_1=0, R34_1=46.66, R44_1=0, R12_2=46.66, R22_2=0, R34_2=46.66, R44_2=0, 
            verbose=True, sigma_integration=3)



    #sigma_z_list = np.linspace(-0.09, 0.09, 100)
    sigma_z_list = np.linspace(-0.2, 0.2, 200)
    longitudinal_density2, rms2, std_dev2,total_luminosity2, peak_pileup_density2, pileup_density2,  mean_z2= longitudinal_luminous_region(sigma_z_list,
            f=11245.5, nb=2452, N1=1.6e11, N2=1.6e11, 
            x_1=0, x_2=0, y_1=0, y_2=0, 
            px_1=0.00018999999999999999, px_2=-0.00018999999999999999, py_1=0, py_2=0, 
            energy_tot1=6.8e12, energy_tot2=6.8e12, 
            deltap_p0_1=0, deltap_p0_2=0, 
            epsilon_x1=2.72930272e-06, epsilon_x2=2.72930272e-06, 
            epsilon_y1= 2.72930272e-06, epsilon_y2=2.72930272e-06, 
            sigma_z1= 0.0761, sigma_z2=0.0761, 
            beta_x1=0.3, beta_x2=0.3, 
            beta_y1=0.3, beta_y2=0.3, 
            alpha_x1=0, alpha_x2=0, 
            alpha_y1=0, alpha_y2=0, 
            dx_1=0, dx_2=0, dy_1=0, dy_2=0, 
            dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0, 
            CC_V_x_1=-3.4e6, CC_f_x_1=400e6, CC_phase_x_1=0, 
            CC_V_x_2=+3.4e6, CC_f_x_2=400e6, CC_phase_x_2=0, 
            CC_V_y_1=-3.4e6, CC_f_y_1=400e6, CC_phase_y_1=0, 
            CC_V_y_2=+3.4e6, CC_f_y_2=400e6, CC_phase_y_2=0, 
            R12_1=46.66, R22_1=0, R34_1=46.66, R44_1=0, R12_2=46.66, R22_2=0, R34_2=46.66, R44_2=0, 
            verbose=True, sigma_integration=3, flag_qGauss_pdf=True, q=3.0/5.0)



    print(rms, std_dev, total_luminosity, peak_pileup_density)
    print(rms2, std_dev2, total_luminosity2, peak_pileup_density2)
    #longitudinal_density,pileup_density




    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(sigma_z_list, longitudinal_density)
    plt.plot(sigma_z_list, longitudinal_density2)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(sigma_z_list, pileup_density)

    print(cmp_FWHM(sigma_z_list, a))
    print(cmp_FWHM(sigma_z_list, a2))
    # %%
