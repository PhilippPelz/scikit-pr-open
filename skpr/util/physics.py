import math as m


def real_space_resolution(E_eV, z, dpix, Npix):
    emass = 510.99906;  # electron rest mass in keV
    hc = 12.3984244;  # h*c
    lam = hc / m.sqrt(E_eV * 1e-3 * (2 * emass + E_eV * 1e-3)) * 1e-10
    res = lam * z
    res /= (dpix*Npix)

    return res


def lam(E_eV):
    emass = 510.99906;  # electron rest mass in keV
    hc = 12.3984244;  # h*c
    lam = hc / m.sqrt(E_eV * 1e-3 * (2 * emass + E_eV * 1e-3))  # in Angstrom
    return lam * 1e-10
