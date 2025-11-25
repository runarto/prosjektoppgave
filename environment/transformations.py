import numpy as np

# WGS-84 ellipsoid constants
WGS84_A = 6378137.0          # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # first eccentricity squared

def eci2ecef(r_eci: np.ndarray, theta_gst: float) -> np.ndarray:
    """Rotate ECI vector into ECEF using Greenwich sidereal angle theta_gst [rad]."""
    c = np.cos(theta_gst)
    s = np.sin(theta_gst)
    R = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [ 0.0, 0.0, 1.0],
    ])
    return R @ r_eci

def ecef2geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic latitude, longitude, and altitude.

    Args:
        r_ecef: ECEF position [m], shape (3,)

    Returns:
        lat: geodetic latitude [rad]
        lon: longitude [rad]
        h:   altitude above WGS-84 ellipsoid [m]
    """
    x, y, z = np.asarray(r_ecef, float).reshape(3)

    # Longitude
    lon = np.arctan2(y, x)

    # Distance from Z axis
    p = np.hypot(x, y)

    # Initial latitude estimate
    # Bowring's method
    a = WGS84_A
    e2 = WGS84_E2
    b = a * np.sqrt(1.0 - e2)

    # Auxiliary angle
    theta = np.arctan2(z * a, p * b)

    st = np.sin(theta)
    ct = np.cos(theta)

    lat = np.arctan2(z + (e2 * b) / (1.0 - e2) * st**3,
                     p - e2 * a * ct**3)

    # Radius of curvature in the prime vertical
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)

    # Altitude
    h = p / np.cos(lat) - N

    return lat, lon, h


def ned2ecef(B_ned: np.ndarray, lat_rad: float, lon_rad: float) -> np.ndarray:
    """Convert magnetic field from NED to ECEF."""
    B_ned = np.asarray(B_ned, float).reshape(3)
    sphi, cphi = np.sin(lat_rad), np.cos(lat_rad)
    slon, clon = np.sin(lon_rad), np.cos(lon_rad)

    R = np.array([
        [-sphi*clon, -slon,       -cphi*clon],
        [-sphi*slon,  clon,       -cphi*slon],
        [ cphi,       0.0,        -sphi     ],
    ])
    return R @ B_ned

def ecef2eci(r_ecef: np.ndarray, theta_gst: float) -> np.ndarray:
    """Rotate ECEF vector into ECI using Greenwich sidereal angle theta_gst [rad]."""
    c = np.cos(theta_gst)
    s = np.sin(theta_gst)
    R = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [ 0.0, 0.0, 1.0],
    ])
    return R @ r_ecef

