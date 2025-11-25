import datetime
from datetime import timezone
import numpy as np

def jd_to_dt(jd: float) -> datetime.datetime:
    """Convert Julian date to a Python datetime object (UTC)."""
    jd_int = int(jd)
    frac = jd - jd_int
    days = jd_int - 1721425  # days since 0001-01-01
    seconds = int(frac * 86400.0 + 0.5)
    dt = datetime.datetime(1, 1, 1) + datetime.timedelta(days=days, seconds=seconds)
    dt = dt.replace(tzinfo=timezone.utc)
    return dt

def gast_from_jd(jd):
    """Greenwich Apparent Sidereal Time in radians."""
    T = (jd - 2451545.0) / 36525.0
    theta = 67310.54841 \
            + (876600.0 * 3600 + 8640184.812866) * T \
            + 0.093104 * T*T \
            - 6.2e-6 * T*T*T
    theta = (theta % 86400) / 86400 * 2*np.pi
    return theta