import numpy as np
import math

def evaluate_over_grid(u, xs, ys, zs):
    xx, yy, zz = np.meshgrid(xs, ys, zs)
    nx, ny, nz = len(xs), len(ys), len(zs)
    x = xx.reshape((1, nx*ny*nz))
    y = yy.reshape((1, nx*ny*nz))
    z = zz.reshape((1, nx*ny*nz))
    points = np.zeros((nx*ny*nz, 3))
    points[:,0] = x
    points[:,1] = y
    points[:,2] = z
    v = evaluate_over_points(u, points)
    n, dims = v.shape
    return v.reshape((nx, ny, nz, dims))

def evaluate_over_points(u, points):
    dims = u.value_size()
    values = np.zeros((len(points), dims))
    for i, x in enumerate(points):
        values[i,:] = u(x)
    return values

def contour_to_points(start, end, contour, npoints):
    ts = np.linspace(start, end, npoints + 1)
    points = np.array([contour(t) for t in ts])
    return points, ts

def evaluate_over_contour(u, start=0, end=2*math.pi, contour=lambda t: [math.cos(t), math.sin(t), 0], npoints=100):
    points, _ = contour_to_points(start, end, contour, npoints)
    return evaluate_over_points(u, points)

def intergrate_over_contour(u, start=0, end=2*math.pi, contour=lambda t: [math.cos(t), math.sin(t), 0], npoints=100):
    dims = u.value_size()
    dt = (end - start)/npoints
    points, ts = contour_to_points(start, end, contour, npoints)
    values = evaluate_over_points(u, points)
    last_point = points[-1]
    extra_point = np.array(contour(end + dt))

    dl = np.diff(points, axis=0)
    dl = np.append(dl, np.array([extra_point - last_point]), axis=0)

    x = numpy.linalg.norm(dl, axis=1)
    y = np.sum(values * dl, axis=1)
    return scipy.integrate.simps(y/x, x * ts / dt)

