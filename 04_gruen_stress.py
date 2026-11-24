"""Read two ODBs and calculate volume-weighted Gruen stress ratios."""
import argparse
import numpy as np

from odbAccess import openOdb
from abaqusConstants import MISES
from odb_utils import aligned, read_mesh, select_step, volumes, write_json


def read_cortical(path, instance_name, step_name):
    odb = openOdb(path=path, readOnly=True)
    try:
        if instance_name not in odb.rootAssembly.instances:
            raise ValueError('Instance %s not found in %s. Available: %s' %
                             (instance_name, path, ', '.join(odb.rootAssembly.instances.keys())))
        instance = odb.rootAssembly.instances[instance_name]
        mesh = read_mesh(instance)
        zones = ['G%d' % i for i in range(1, 8)]
        missing = [zone for zone in zones if zone not in instance.elementSets]
        if missing:
            raise ValueError('Required cortical element sets are missing: %s. '
                             'Define G1-G7 in the ODB; available instance sets: %s' %
                             (', '.join(missing), ', '.join(instance.elementSets.keys())))
        membership = {}
        for zone in zones:
            labels = np.array([element.label for element in instance.elementSets[zone].elements],
                              dtype=np.int64)
            membership[zone] = np.unique(labels)
            if not len(labels):
                raise ValueError(zone + ' is empty.')
        frame = select_step(odb, step_name).frames[-1]
        if 'S' not in frame.fieldOutputs:
            raise ValueError('Final frame has no S stress output in ' + path)
        stress = frame.fieldOutputs['S'].getScalarField(invariant=MISES)
        vm = aligned(stress, instance, mesh['labels'])
        if not np.isfinite(vm).all():
            raise ValueError('Finite integration-point stresses are required in ' + path)
        return mesh, membership, vm
    finally:
        odb.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='Intact ODB')
    parser.add_argument('--implant', required=True, help='Implanted ODB')
    parser.add_argument('--reference-instance', default='CORTICAL')
    parser.add_argument('--implant-instance', default='CORTICAL')
    parser.add_argument('--step', help='Static step name; default: last step')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    base, base_zones, intact_vm = read_cortical(
        args.reference, args.reference_instance, args.step)
    implant, implant_zones, implanted_vm = read_cortical(
        args.implant, args.implant_instance, args.step)
    for key in ('nodes', 'labels', 'conn'):
        if not np.array_equal(base[key], implant[key]):
            raise ValueError('Intact and implanted cortical meshes differ: ' + key)
    if not np.allclose(base['xyz'], implant['xyz'], rtol=0, atol=1.e-4):
        raise ValueError('Intact and implanted cortical node coordinates differ.')

    indices = {}
    labels = base['labels']
    for zone in base_zones:
        if not np.array_equal(base_zones[zone], implant_zones[zone]):
            raise ValueError('Intact and implanted region membership differs: ' + zone)
        ids = np.searchsorted(labels, base_zones[zone])
        if np.any(ids >= len(labels)) or not np.array_equal(labels[ids], base_zones[zone]):
            raise ValueError(zone + ' contains unknown cortical element labels.')
        indices[zone] = ids
    # Pool the elements once; do not average the seven zone means or ratios.
    indices['G1_G7_UNION'] = np.unique(np.concatenate(list(indices.values())))
    reference_volumes = volumes(base)
    regions = {}
    for zone, ids in indices.items():
        weights = reference_volumes[ids]
        total_volume = float(weights.sum())
        mean = float(np.dot(implanted_vm[ids], weights) / total_volume)
        intact_mean = float(np.dot(intact_vm[ids], weights) / total_volume)
        if intact_mean <= 0:
            raise ValueError('Positive intact mean required for the stress ratio in ' + zone)
        regions[zone] = {
            'elements': len(ids),
            'reference_volume_mm3': total_volume,
            'VM_mean_MPa': mean,
            'Intact_VM_mean_MPa': intact_mean,
            'stress_ratio': mean / intact_mean,
        }
    write_json(args.output, {'regions': regions})


if __name__ == '__main__':
    main()
