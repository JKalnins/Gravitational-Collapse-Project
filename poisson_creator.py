import numpy as np

class idiot(Exception):  # this is very useful
    pass


def any_is_close(pt, pts, mindist):
    """
    checks if any points in pts are <= mindist from pt
    inputs:
        pt: 1xm array (m-dimensional space)
        pts: Nxm array
        mindist: float
    Outputs:
        True if any pts are closer than mindist to pt
            False otherwise
    """
    npts = len(pts[:, 0])
    dxs = np.array([pts[i, :] - pt for i in range(npts)])
    drs = np.array([np.sqrt(np.sum(dxs[i, :] ** 2)) for i in range(npts)])
    for dr in drs:
        if dr <= mindist:
            return True
    return False


def poisson_distributor(n, rmax, dmin, dims, prevs = [], report = True):
    """
    Creates an array of n points in a sphere of radius rmax at least dmin
    away from one another. Can be done in 2 or 3D but 2D returns an nx3 array
    with zeros in the 3rd column
    Inputs:
        n: int
        rmax: float
        dmin: float
        dims: int (=< 3 only)
    Outputs:
        pts: nx3 array
    """
    if prevs != []:
        prev_l = len(prevs[:,0])
        n = n + prev_l
        pts = np.zeros((n,3))
        pts[:prev_l,:] = prevs
        i = prev_l
    else:
        i = 0
        pts = np.zeros((n,3))
    count = 0
    while i < n:  # creating the list: I know this is inefficient (too many checks)
        if count > 50 * n:
            print("too many points you idiot")
            raise idiot
        pos = np.random.uniform(-rmax, rmax, size=dims)
        r = np.sqrt(np.sum(pos ** 2))
        check = True
        if r >= rmax:
            check = False
        if i != 0 and check:
            if not any_is_close(pos, pts[0:i, 0:dims], dmin):
                pts[i, 0:dims] = pos
                i += 1
        elif i == 0:
            pts[i, 0:dims] = pos
            i += 1
        count += 1
    if report:
        print("counting complete")
    return(pts)