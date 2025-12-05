from environment.transformations import eci2ecef, ecef2geodetic, ned2ecef, ecef2eci
import numpy as np
from environment.utils import jd_to_dt, gast_from_jd
import pyIGRF as igrf
from sgp4.api import Satrec, WGS84, jday
from skyfield.api import load


class EnvironmentModel:
    
    def __init__(self):
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self.sun = self.eph['sun']
        self.earth = self.eph['earth']
    
    def get_B_eci(self, r_eci: np.ndarray, jd: float) -> np.ndarray:
        dt = jd_to_dt(jd)
        theta_gst=gast_from_jd(jd)
        r_ecef = eci2ecef(r_eci=r_eci, theta_gst=theta_gst)
        lat, lon, h = ecef2geodetic(r_ecef)
        _, _, _, N, E, D, _ = igrf.igrf_value(lat=np.degrees(lat),
                                 lon=np.degrees(lon),
                                 alt=h/1000.0,
                                 year=dt.year + (dt.timetuple().tm_yday - 1) / 365.25)
        
        B_ned = np.array([N, E, D]) 
        B_ecef = ned2ecef(B_ned, lat, lon)
        B_eci = ecef2eci(B_ecef, theta_gst=theta_gst)
        return B_eci / np.linalg.norm(B_eci)

    def get_sun_eci(self, jd: float) -> np.ndarray:
        t = self.ts.tdb(jd=jd)
        r = self.sun.at(t).position.km - self.earth.at(t).position.km
        r = np.asarray(r)
        return r / np.linalg.norm(r)


class OrbitEnvironmentModel(EnvironmentModel):
    def __init__(self):
        super().__init__()
        line1 = "1 60531U 24149BR  25329.08965914  .00005754  00000-0  50150-3 0  9993"
        line2 = "2 60531  97.7030  43.3117 0001033  94.6506 265.4834 14.97264647 69499"
        self.sat = Satrec.twoline2rv(line1, line2)

    def get_r_eci(self, jd: float) -> np.ndarray:
        jd_int = float(np.floor(jd))
        fr = jd - jd_int
        e, r_km, _ = self.sat.sgp4(jd_int, fr)
        if e != 0:
            raise RuntimeError(f"SGP4 error code {e}")
        return np.asarray(r_km) * 1000.0  # meters