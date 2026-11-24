"""Shared ODB field reading and C3D4 geometry for the six analysis scripts."""
import json
from pathlib import Path
import numpy as np

CHUNK = 50000


def select_step(odb, name=None):
    step = odb.steps[name] if name else list(odb.steps.values())[-1]
    if str(step.procedure).strip().upper() not in ('*STATIC', '*STATIC, DIRECT'):
        raise ValueError('Select a static loading step.')
    if not len(step.frames):
        raise ValueError('The selected step has no saved frames.')
    if abs(step.frames[-1].frameValue-step.timePeriod) > max(1.e-8, abs(step.timePeriod)*1.e-6):
        raise ValueError('The selected loading step has not completed.')
    return step


def read_field(field, instance=None, nodal=False):
    """Return sorted labels and copied scalar/vector values from one instance."""
    subset = field.getSubset(region=instance) if instance is not None else field
    expected = 'NODAL' if nodal else 'INTEGRATION_POINT'
    labels, arrays = [], []
    fallback = False
    for block in subset.bulkDataBlocks:
        if str(block.position) != expected:
            raise ValueError(field.name + ' must be stored at ' + expected)
        ids = np.asarray(block.nodeLabels if nodal else block.elementLabels, dtype=np.int64)
        if not len(ids):
            continue
        try:
            raw = block.data
        except Exception:
            fallback = True
            break
        labels.append(ids.copy())
        arrays.append(np.asarray(raw, dtype=float).reshape(len(ids), -1).copy())
    if fallback:
        ids, data = [], []
        for value in subset.values:
            if str(value.position) != expected:
                raise ValueError(field.name + ' must be stored at ' + expected)
            ids.append(value.nodeLabel if nodal else value.elementLabel)
            try:
                raw = value.data
            except Exception:
                raw = value.dataDouble
            data.append(np.asarray(raw, dtype=float).reshape(-1))
        labels = [np.asarray(ids, dtype=np.int64)]
        arrays = [np.asarray(data, dtype=float)]
    if not labels or not sum(len(ids) for ids in labels):
        raise ValueError('No field values found: ' + field.name)
    labels, data = np.concatenate(labels), np.concatenate(arrays)
    order = np.argsort(labels, kind='stable')
    labels, data = labels[order], data[order]
    if instance is not None and np.any(np.diff(labels) <= 0):
        raise ValueError('Expected one value per node/C3D4 element: ' + field.name)
    return labels, data[:, 0] if data.shape[1] == 1 else data


def aligned(field, instance, wanted, nodal=False):
    labels, values = read_field(field, instance, nodal)
    wanted = np.asarray(wanted, dtype=np.int64)
    index = np.searchsorted(labels, wanted)
    if np.any(index >= len(labels)) or not np.array_equal(labels[index], wanted):
        raise ValueError('Required node/element labels missing from ' + field.name)
    return values[index]


def read_mesh(instance):
    nodes = np.array([n.label for n in instance.nodes], dtype=np.int64)
    xyz = np.array([n.coordinates for n in instance.nodes], dtype=float)
    if not len(nodes) or not len(instance.elements):
        raise ValueError('The selected mesh is empty.')
    if any(e.type != 'C3D4' for e in instance.elements):
        raise ValueError('These scripts require C3D4 elements.')
    labels = np.array([e.label for e in instance.elements], dtype=np.int64)
    conn = np.array([e.connectivity for e in instance.elements], dtype=np.int64)
    ni, ei = np.argsort(nodes), np.argsort(labels)
    nodes, xyz, labels, conn = nodes[ni], xyz[ni], labels[ei], conn[ei]
    if np.any(np.diff(nodes) <= 0) or np.any(np.diff(labels) <= 0) or not np.isfinite(xyz).all():
        raise ValueError('Invalid mesh labels or coordinates.')
    index = np.searchsorted(nodes, conn)
    if np.any(index >= len(nodes)) or not np.array_equal(nodes[index], conn):
        raise ValueError('Unknown connectivity nodes.')
    return dict(nodes=nodes, xyz=xyz, labels=labels, conn=conn, index=index)


def signed_volumes(xyz, index):
    result = np.empty(len(index))
    for start in range(0, len(index), CHUNK):
        stop = min(start+CHUNK, len(index))
        p = xyz[index[start:stop]]
        result[start:stop] = np.einsum('ij,ij->i', p[:, 1]-p[:, 0],
            np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0])) / 6.
    if not np.isfinite(result).all() or np.any(result == 0):
        raise ValueError('Invalid tetrahedron volume.')
    return result


def volumes(mesh):
    return np.abs(signed_volumes(mesh['xyz'], mesh['index']))


def reaction_load(frame):
    """Load magnitude from the vector sum of all saved nodal reaction forces."""
    if 'RF' not in frame.fieldOutputs:
        raise ValueError('RF field output is required for actual load values.')
    _, forces = read_field(frame.fieldOutputs['RF'], nodal=True)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.isfinite(forces).all():
        raise ValueError('RF must contain finite three-component nodal vectors.')
    return float(np.linalg.norm(forces.sum(axis=0)))


def write_json(path, data):
    path = Path(path)
    if path.suffix.lower() != '.json':
        raise ValueError('Use a .json output filename.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
