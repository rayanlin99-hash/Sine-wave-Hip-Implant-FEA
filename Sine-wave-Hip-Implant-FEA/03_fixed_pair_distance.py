"""Follow fixed implant/bone surface material points directly from an ODB.

At each height, select the lateral/medial bone intersection with X=0,
then the closest implant point and its closest bone point. Keep those
triangle weights fixed through loading. Distance includes initial separation.
The selection/interpolation follows the original six-point analysis; no
corotation, contact-dependent reselection, CSLIP or COPEN is used.
"""
import argparse
from pathlib import Path
import numpy as np
from odb_utils import aligned, reaction_load, select_step, write_json

CHUNK = 50000
FACE_NODES = {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)}


def surface_mesh(instance, surface_name, initial_frame):
    """Extract C3D4 faces and orient them away from the parent tetrahedra."""
    if surface_name not in instance.surfaces:
        raise ValueError('Surface %s is missing on %s.' % (surface_name, instance.name))
    surface = instance.surfaces[surface_name]
    if not len(surface.elements) or len(surface.elements) != len(surface.faces):
        raise ValueError('An element-face surface is required: ' + surface_name)
    triangles, opposite = [], []
    for element, face in zip(surface.elements, surface.faces):
        if element.type != 'C3D4':
            raise ValueError('The selected surfaces must use C3D4 elements.')
        face_number = int(str(face).replace('FACE', ''))
        slots = FACE_NODES[face_number]
        triangles.append([element.connectivity[j] for j in slots])
        opposite.append(element.connectivity[next(j for j in range(4) if j not in slots)])
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.unique(np.r_[triangles.ravel(), opposite])
    node_labels = np.array([node.label for node in instance.nodes], dtype=np.int64)
    coordinates = np.array([node.coordinates for node in instance.nodes], dtype=float)
    order = np.argsort(node_labels)
    node_labels, coordinates = node_labels[order], coordinates[order]
    index = np.searchsorted(node_labels, labels)
    if np.any(index >= len(node_labels)) or not np.array_equal(node_labels[index], labels):
        raise ValueError('Surface connectivity contains unknown nodes.')
    xyz = coordinates[index]
    u0 = aligned(initial_frame.fieldOutputs['U'], instance, labels, nodal=True)
    if u0.shape != xyz.shape or not (np.isfinite(xyz).all() and np.isfinite(u0).all()):
        raise ValueError('Surface coordinates and initial U must be finite 3D vectors.')
    triangles = np.searchsorted(labels, triangles)
    opposite = np.searchsorted(labels, opposite)
    initial = xyz + u0
    faces = initial[triangles]
    normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError('Degenerate surface triangle.')
    reverse = np.einsum('ij,ij->i', normals, initial[opposite]-faces[:, 0]) > 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    return dict(labels=labels, xyz=xyz, u0=u0, triangles=triangles,
                initial_triangles=initial[triangles])


def ray_points(triangles, x, z):
    """Intersect the X=x, Z=z line with actual surface triangles."""
    points = []
    for start in range(0, len(triangles), CHUNK):
        t = triangles[start:start+CHUNK]
        a = t[:, 0][:, [0, 2]]
        b = (t[:, 1]-t[:, 0])[:, [0, 2]]
        c = (t[:, 2]-t[:, 0])[:, [0, 2]]
        q = np.array([x, z])-a
        den = b[:, 0]*c[:, 1]-b[:, 1]*c[:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (q[:, 0]*c[:, 1]-q[:, 1]*c[:, 0])/den
            w = (b[:, 0]*q[:, 1]-b[:, 1]*q[:, 0])/den
            good = (abs(den)>1e-12) & (v>=-1e-7) & (w>=-1e-7) & (v+w<=1+1e-7)
        ids = np.flatnonzero(good)
        weights = np.column_stack([1-v[ids]-w[ids], v[ids], w[ids]])
        points.append(np.einsum('ni,nij->nj', weights, t[ids]))
    return np.concatenate(points)


def closest_point(point, triangles):
    """Exact closest point on the supplied triangles, with bounded memory."""
    best_distance = np.inf
    best_point, best_face = None, None
    for start in range(0, len(triangles), CHUNK):
        t = triangles[start:start+CHUNK]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab, ac, ap = b-a, c-a, point-a
        dot = lambda x, y: np.einsum('ij,ij->i', x, y)
        d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
        d20, d21 = dot(ap, ab), dot(ap, ac)
        den = d00*d11-d01*d01
        if np.any(den <= 0):
            raise ValueError('Degenerate surface triangle.')
        v = (d11*d20-d01*d21)/den
        w = (d00*d21-d01*d20)/den
        projected = a+v[:, None]*ab+w[:, None]*ac
        inside = (v>=-1e-12) & (w>=-1e-12) & (v+w<=1+1e-12)
        distance2 = np.where(inside, dot(point-projected, point-projected), np.inf)
        closest = projected.copy()
        for p, q in ((a, b), (b, c), (c, a)):
            edge = q-p
            fraction = np.clip(dot(point-p, edge)/dot(edge, edge), 0, 1)
            candidate = p+fraction[:, None]*edge
            d2 = dot(point-candidate, point-candidate)
            take = d2 < distance2
            closest[take], distance2[take] = candidate[take], d2[take]
        hit = int(np.argmin(distance2))
        if distance2[hit] < best_distance:
            best_distance = distance2[hit]
            best_point, best_face = closest[hit].copy(), start+hit
    return best_point, best_face


def barycentric(point, triangle):
    uv = np.linalg.lstsq((triangle[1:]-triangle[0]).T, point-triangle[0], rcond=None)[0]
    weights = np.array([1-uv.sum(), uv[0], uv[1]])
    if not (weights.min()>-1e-7 and weights.max()<1+1e-7) or np.linalg.norm(weights@triangle-point)>=1e-7:
        raise ValueError('Invalid material-point interpolation weights.')
    return weights


def segment_obstructions(start, end, triangles):
    """Reject a paired segment passing through another implant face."""
    count = 0
    d = end-start
    for offset in range(0, len(triangles), CHUNK):
        t = triangles[offset:offset+CHUNK]
        e1, e2 = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
        pv = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, pv)
        tv = start-t[:, 0]
        qv = np.cross(tv, e1)
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.einsum('ij,ij->i', tv, pv)/det
            v = (qv@d)/det
            s = np.einsum('ij,ij->i', e2, qv)/det
            hit = (abs(det)>1e-12) & (u>=-1e-9) & (v>=-1e-9) & (u+v<=1+1e-9) & (s>1e-5) & (s<1-1e-5)
        count += int(hit.sum())
    return count


def select_pairs(implant, bone, heights):
    pairs = []
    it, bt = implant['initial_triangles'], bone['initial_triangles']
    for side in ('lateral', 'medial'):
        for z in heights:
            intersections = ray_points(bt, 0.0, z)
            if not len(intersections):
                raise ValueError('No bone surface intersection at X=0, Z=%g.' % z)
            index = np.argmin(intersections[:, 1]) if side == 'lateral' else np.argmax(intersections[:, 1])
            anchor = intersections[index]
            ip, ai = closest_point(anchor, it)
            bp, bi = closest_point(ip, bt)
            if segment_obstructions(ip, bp, it):
                raise ValueError('Obstructed implant-bone pairing at %s Z=%g.' % (side, z))
            normal = np.cross(bt[bi, 1]-bt[bi, 0], bt[bi, 2]-bt[bi, 0])
            if not (normal[1]>0 if side == 'lateral' else normal[1]<0):
                raise ValueError('Bone surface has the wrong cavity-facing direction at %s Z=%g.' % (side, z))
            pair = {'point_id':('L' if side == 'lateral' else 'M')+('%g' % z),
                    'side':side, 'nominal_Z_mm':float(z), 'anchor_XYZ_mm':anchor.tolist()}
            for role, surface, face, point in (('implant', implant, ai, ip), ('bone', bone, bi, bp)):
                ids = surface['triangles'][face]
                pair[role+'_node_labels'] = surface['labels'][ids].tolist()
                pair[role+'_face_XYZ_mm'] = surface['xyz'][ids].tolist()
                pair[role+'_weights'] = barycentric(point, surface['initial_triangles'][face]).tolist()
            pairs.append(pair)
    return pairs


def run(args):
    from odbAccess import openOdb
    odb = openOdb(path=str(Path(args.odb).resolve()), readOnly=True)
    try:
        step = select_step(odb, args.step)
        if abs(step.frames[0].frameValue)>1e-10:
            raise ValueError('The selected step must include its initial zero-time frame.')
        instances = {'implant':odb.rootAssembly.instances[args.implant_instance],
                     'bone':odb.rootAssembly.instances[args.bone_instance]}
        surfaces = {role:surface_mesh(instances[role], getattr(args, role+'_surface'), step.frames[0])
                    for role in ('implant', 'bone')}
        pairs = select_pairs(surfaces['implant'], surfaces['bone'], args.heights)
        wanted = {role:np.unique([n for p in pairs for n in p[role+'_node_labels']])
                  for role in ('implant', 'bone')}
        history = []
        for fi, frame in enumerate(step.frames):
            displacement = {}
            for role in ('implant', 'bone'):
                if fi == 0:
                    ids = np.searchsorted(surfaces[role]['labels'], wanted[role])
                    displacement[role] = surfaces[role]['u0'][ids]
                else:
                    displacement[role] = aligned(frame.fieldOutputs['U'], instances[role], wanted[role], nodal=True)
                if displacement[role].shape != (len(wanted[role]), 3) or not np.isfinite(displacement[role]).all():
                    raise ValueError('Selected U must contain finite three-component vectors.')
            load = reaction_load(frame)
            for p in pairs:
                positions = {}
                for role in ('implant', 'bone'):
                    ids = np.searchsorted(wanted[role], p[role+'_node_labels'])
                    xyz = np.array(p[role+'_face_XYZ_mm']) + displacement[role][ids]
                    positions[role] = np.einsum('i,ij->j', np.array(p[role+'_weights']), xyz)
                distance = float(np.linalg.norm(positions['implant']-positions['bone'])*1000)
                history.append({'point_id':p['point_id'], 'frame_index':fi,
                                'step_time':float(frame.frameValue), 'load_N':load, 'distance_um':distance})
        summary = []
        for p in pairs:
            values = [row['distance_um'] for row in history if row['point_id']==p['point_id']]
            summary.append({'point_id':p['point_id'], 'initial_distance_um':values[0], 'final_distance_um':values[-1]})
        report = {'source_odb':str(Path(args.odb).resolve()), 'step':step.name,
                  'implant_instance':args.implant_instance, 'bone_instance':args.bone_instance,
                  'implant_surface':args.implant_surface, 'bone_surface':args.bone_surface,
                  'metric':'Current 3D distance between fixed material points; initial separation retained',
                  'load':'Magnitude of the assembly-summed nodal RF vector',
                  'units':{'coordinates':'mm', 'distance':'um', 'load':'N'},
                  'sampling':pairs, 'history':history, 'summary':summary}
    finally:
        odb.close()
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odb', required=True)
    parser.add_argument('--implant-instance', required=True)
    parser.add_argument('--bone-instance', required=True)
    parser.add_argument('--implant-surface', default='MIDDLE')
    parser.add_argument('--bone-surface', default='MIDDLE')
    parser.add_argument('--heights', nargs='+', type=float, default=[40.0, 70.0])
    parser.add_argument('--step', help='Default: last completed static step')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not np.isfinite(args.heights).all() or len(set(args.heights)) != len(args.heights):
        raise ValueError('Heights must be finite and unique.')
    run(args)


if __name__ == '__main__':
    main()
