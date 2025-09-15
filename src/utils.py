import numpy as np

def vec2(v):
    return np.array(v, dtype=np.float32).reshape(2, )


def length2(v):
    return float(np.linalg.norm(v))


def normalize2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def perp(v):
    return np.array([v[1], -v[0]], dtype=np.float32)


def barycentric_coords(p, a, b, c):
    # 2D barycentric
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float32)
