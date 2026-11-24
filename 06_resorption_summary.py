"""Summarise RESORP_S035 directly from a resorption output ODB.
All supplied cortical elements, including zero-rate elements, enter the denominator.
"""
import argparse
import numpy as np

from odbAccess import openOdb
from odb_utils import aligned, read_mesh, volumes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odb', required=True, help='Resorption output ODB')
    parser.add_argument('--instance', default='CORTICAL')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    odb = openOdb(path=args.odb, readOnly=True)
    try:
        if args.instance not in odb.rootAssembly.instances:
            raise ValueError('Instance %s not found. Available: %s' %
                             (args.instance, ', '.join(odb.rootAssembly.instances.keys())))
        instance = odb.rootAssembly.instances[args.instance]
        mesh = read_mesh(instance)
        step_names = list(odb.steps.keys())
        if not step_names or not odb.steps[step_names[-1]].frames:
            raise ValueError('ODB contains no result frame.')
        frame = odb.steps[step_names[-1]].frames[-1]
        if 'RESORP_S035' not in frame.fieldOutputs:
            raise ValueError('Last frame has no RESORP_S035 output. Run script 05 first.')
        rate = aligned(frame.fieldOutputs['RESORP_S035'], instance, mesh['labels'])
        if not np.isfinite(rate).all() or np.any(rate > 0):
            raise ValueError('Initial resorption rates must be finite and nonpositive.')
        reference_volumes = volumes(mesh)
    finally:
        odb.close()

    negative = rate < 0
    total_volume = float(np.sum(reference_volumes, dtype=np.float64))
    negative_volume = float(np.sum(reference_volumes[negative], dtype=np.float64))
    weighted_sum = float(np.sum(reference_volumes * rate, dtype=np.float64))
    # Both denominators include zero-rate elements.
    report = {
        'rate_units': 'kg m^-3 ctu^-1',
        'volume_units': 'mm^3',
        'total_volume_mm3': total_volume,
        'negative_volume_mm3': negative_volume,
        'mean_rate': weighted_sum / total_volume,
        'resorption_percent': negative_volume / total_volume * 100,
    }
    write_json(args.output, report)


if __name__ == '__main__':
    main()
