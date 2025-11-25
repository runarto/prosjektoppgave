import numpy as np
import yaml

def get_skew_matrix(v: np.ndarray[3]) -> np.ndarray[3, 3]:
    """Get the cross product matrix [v×] for a 3D vector v."""
    v = np.asarray(v, float).reshape(3)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    
    
def load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dictionary"""
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data

