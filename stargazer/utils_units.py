from __future__ import annotations
import math

G = 6.67430e-11
M_sun_kg = 1.98847e30
M_jup_kg = 1.89813e27
DAY_S = 86400.0

def msun_to_kg(m_sun: float) -> float:
    return m_sun * M_sun_kg

def mjup_to_kg(m_jup: float) -> float:
    return m_jup * M_jup_kg

def kg_to_mjup(m_kg: float) -> float:
    return m_kg / M_jup_kg

def days_to_seconds(days: float) -> float:
    return days * DAY_S

def seconds_to_days(sec: float) -> float:
    return sec / DAY_S

def kepler_semi_major_axis_from_period(P_s: float, M_star_kg: float, m_planet_kg: float = 0.0) -> float:
    mu = G * (M_star_kg + m_planet_kg)
    a_cubed = mu * (P_s / (2.0 * math.pi))**2
    return a_cubed ** (1.0/3.0)

def semi_amplitude_ms(m_sin_i_mjup: float, P_days: float, e: float, M_star_sun: float) -> float:
    """Approximate RV semi-amplitude K [m/s].
    K ≈ 28.4329 m/s * (m_p sin i / M_Jup) / (M_star/M_sun)^{2/3} (P/1yr)^{-1/3} / sqrt(1-e^2)

    Reference: Perryman (2018) *The Exoplanet Handbook*, Eq. 5.1
    """
    if m_sin_i_mjup < 0:
        raise ValueError(f"m_sin_i must be ≥0, got {m_sin_i_mjup}")
    if P_days <= 0:
        raise ValueError(f"Period must be >0 days, got {P_days}")
    if not (0.0 <= e < 1.0):
        raise ValueError(f"Eccentricity must be in [0,1), got {e}")
    if M_star_sun <= 0:
        raise ValueError(f"Stellar mass must be >0, got {M_star_sun} M_sun")
    P_years = P_days / 365.25
    return 28.4329 * m_sin_i_mjup * (M_star_sun ** (-2.0/3.0)) * (P_years ** (-1.0/3.0)) / math.sqrt(1.0 - e*e)
