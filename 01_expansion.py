"""Read a C3D4 ODB and calculate expansion at 21 reference heights."""

import argparse
from pathlib import Path

import numpy as np
from odbAccess import openOdb

from odb_utils import aligned, read_mesh, select_step, write_json

HEIGHTS = np.linspace(27.0, 77.0, 21)


def section_points(x, u, connectivity, z, zmin, zmax):
    """Select the section extrema and interpolate their displacements."""
    active = np.flatnonzero((zmin <= z) & (zmax >= z))
    connectivity = connectivity[active]
    best = {}
    for ia, ib in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        na, nb = connectivity[:, ia], connectivity[:, ib]
        pa, pb = x[na], x[nb]
        dz = pb[:, 2] - pa[:, 2]
        valid = abs(dz) > 1e-12
        fraction = np.zeros(len(dz))
        fraction[valid] = (z - pa[valid, 2]) / dz[valid]
        valid &= (fraction >= -1e-10) & (fraction <= 1 + 1e-10)
        rows = np.flatnonzero(valid)
        if not len(rows):
            continue
        weights = fraction[rows]
        points = pa[rows] * (1 - weights[:, None]) + pb[rows] * weights[:, None]
        for side, extreme in [('lateral', points[:, 1].min()), ('medial', points[:, 1].max())]:
            ties = np.flatnonzero(abs(points[:, 1] - extreme) < 1e-8)
            chosen = int(ties[np.argmin(abs(points[ties, 0]))])
            edge = int(rows[chosen])
            point = points[chosen]
            if side not in best:
                improve, tie = True, False
            else:
                previous = best[side][0]
                if side == 'lateral':
                    improve = point[1] < previous[1] - 1e-8
                else:
                    improve = point[1] > previous[1] + 1e-8
                tie = abs(point[1] - previous[1]) < 1e-8 and abs(point[0]) < abs(previous[0])
            if improve or tie:
                displacement = u[na[edge]] * (1 - fraction[edge]) + u[nb[edge]] * fraction[edge]
                best[side] = (point, displacement)
    if len(best) != 2 or not best['lateral'][0][1] < 0 < best['medial'][0][1]:
        raise ValueError('No opposed material points straddling Y=0 at Z=%.6g mm.' % z)
    points = np.array([best[side][0] for side in ('lateral', 'medial')])
    displacements = np.array([best[side][1] for side in ('lateral', 'medial')])
    return points, displacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odb', required=True)
    parser.add_argument('--instance', required=True, help='Implant instance name')
    parser.add_argument('--step', help='Static step; defaults to the last step')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    path = str(Path(args.odb).resolve())
    odb = openOdb(path=path, readOnly=True)
    try:
        if args.instance not in odb.rootAssembly.instances:
            raise ValueError('Instance not found: ' + args.instance)
        instance = odb.rootAssembly.instances[args.instance]
        step = select_step(odb, args.step)
        step_name = step.name
        frame_time = float(step.frames[-1].frameValue)
        mesh = read_mesh(instance)
        x = mesh['xyz']
        connectivity = mesh['index']
        u = aligned(step.frames[-1].fieldOutputs['U'], instance, mesh['nodes'], nodal=True)
        u0 = aligned(step.frames[0].fieldOutputs['U'], instance, mesh['nodes'], nodal=True)
    finally:
        odb.close()

    if u.shape != x.shape or u0.shape != x.shape:
        raise ValueError('Each node must have three displacement components.')
    if not (np.isfinite(x).all() and np.isfinite(u).all() and np.isfinite(u0).all()):
        raise ValueError('Coordinates and displacements must be finite.')
    if np.max(abs(u0)) >= 1e-8:
        raise ValueError('The method requires zero initial displacement.')

    zmin = np.full(len(connectivity), np.inf)
    zmax = np.full(len(connectivity), -np.inf)
    for column in range(4):
        z = x[connectivity[:, column], 2]
        np.minimum(zmin, z, out=zmin)
        np.maximum(zmax, z, out=zmax)

    rows = []
    for z in HEIGHTS:
        initial, displacement = section_points(x, u, connectivity, float(z), zmin, zmax)
        if np.max(abs(initial[:, 2] - z)) >= 1e-8:
            raise ValueError('Section points do not match the requested reference height.')
        final = initial + displacement
        d0 = float(np.linalg.norm(initial[1] - initial[0]))
        d1 = float(np.linalg.norm(final[1] - final[0]))
        rows.append({
            'Z_mm': float(z),
            'initial_distance_mm': d0,
            'final_distance_mm': d1,
            'distance_change_um': (d1 - d0) * 1000,
        })
    report = {
        'odb': path,
        'instance': args.instance,
        'step': step_name,
        'frame_time': frame_time,
        'units': {'coordinates': 'mm', 'distance_change': 'um'},
        'selection': 'Reference-Z sections; minimum and maximum Y; X is unrestricted',
        'rows': rows,
    }
    write_json(args.output, report)


if __name__ == '__main__':
    main()
