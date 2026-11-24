"""Read contact pressure and same-node resultant slip from every saved ODB frame."""

import argparse
from pathlib import Path

import numpy as np
from odbAccess import openOdb

from odb_utils import reaction_load, read_field, select_step, write_json


def valid(values):
    return np.isfinite(values) & (values > -1e30) & (values < 1e30)


def contact_keys(frame, pair=None):
    """Use the same exact contact-pair suffix for all three fields."""
    pressure_keys = {}
    for key in frame.fieldOutputs.keys():
        parts = key.split(None, 1)
        if parts[0] == 'CPRESS':
            suffix = parts[1].strip() if len(parts) == 2 else ''
            pressure_keys[suffix] = key
    if pair is None:
        if len(pressure_keys) != 1:
            raise ValueError('Specify --pair using one of the CPRESS suffixes: '
                             + ', '.join(sorted(pressure_keys)))
        pair = next(iter(pressure_keys))
    if pair not in pressure_keys:
        raise ValueError('CPRESS contact pair not found: ' + pair)

    keys = {}
    for name in ('CPRESS', 'CSLIP1', 'CSLIP2'):
        matches = []
        for key in frame.fieldOutputs.keys():
            parts = key.split(None, 1)
            suffix = parts[1].strip() if len(parts) == 2 else ''
            if parts[0] == name and suffix == pair:
                matches.append(key)
        if len(matches) != 1:
            raise ValueError('Expected one ' + name + ' field for contact pair ' + pair)
        keys[name] = matches[0]
    return pair, keys


def summary(values, labels, instance, mask=None):
    selected = valid(values)
    if mask is not None:
        selected &= mask
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        return {'count': 0}
    maximum = indices[np.argmax(values[indices])]
    label = int(labels[maximum])
    return {
        'count': int(len(indices)),
        'max': float(values[maximum]),
        'max_node': label,
        'max_xyz': [float(value) for value in instance.getNodeFromLabel(label).coordinates],
    }


def frame_metrics(frame, instance, pair):
    _, keys = contact_keys(frame, pair)
    arrays = {}
    labels = None
    for name, key in keys.items():
        current_labels, values = read_field(frame.fieldOutputs[key], instance, nodal=True)
        if values.ndim != 1:
            raise ValueError(name + ' must contain one scalar per contact node.')
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError('CPRESS and CSLIP fields must report the same nodes for the same pair.')
        arrays[name] = values

    pressure = arrays['CPRESS']
    slip1, slip2 = arrays['CSLIP1'], arrays['CSLIP2']
    active = valid(pressure) & (pressure > 1e-6)
    slip = np.hypot(slip1, slip2)
    slip[~(valid(slip1) & valid(slip2))] = np.nan
    return {
        'reported_contact_nodes': len(labels),
        'positive_pressure_nodes': int(active.sum()),
        'CPRESS': {
            'all_reported': summary(pressure, labels, instance),
            'positive_pressure_nodes': summary(pressure, labels, instance, active),
        },
        'tangential_slip_mm': {
            'all_reported': summary(slip, labels, instance),
            'positive_pressure_nodes': summary(slip, labels, instance, active),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odb', required=True)
    parser.add_argument('--instance', required=True, help='Secondary implant instance name')
    parser.add_argument('--pair', help='Exact suffix after CPRESS; required when several pairs exist')
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
        pair, _ = contact_keys(step.frames[-1], args.pair)
        frame_history = []
        for index, frame in enumerate(step.frames):
            metrics = frame_metrics(frame, instance, pair)
            frame_history.append({
                'frame_index': index,
                'time': float(frame.frameValue),
                'load_N': reaction_load(frame),
                'CPRESS_max_MPa': metrics['CPRESS']['all_reported'].get('max'),
                'positive_pressure_nodes': metrics['positive_pressure_nodes'],
                'slip_max_mm': metrics['tangential_slip_mm']['all_reported'].get('max'),
                'slip_max_positive_pressure_mm': metrics['tangential_slip_mm']['positive_pressure_nodes'].get('max'),
            })
        report = {
            'odb': path,
            'instance': args.instance,
            'step': step.name,
            'pair': pair,
            'positive_pressure_threshold_MPa': 1e-6,
            'units': {'CPRESS': 'MPa', 'CSLIP': 'mm', 'coordinates': 'mm', 'load': 'N'},
            **metrics,
            'frame_history': frame_history,
        }
    finally:
        odb.close()
    write_json(args.output, report)


if __name__ == '__main__':
    main()
