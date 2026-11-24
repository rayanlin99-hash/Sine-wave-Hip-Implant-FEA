"""Initial cortical resorption relative to intact bone, s=0.35.
N-mm-MPa inputs; matching C3D4 meshes, materials and loading.
Rate = B*min(S_implant-(1-s)*S_intact, 0), in kg/m^3/ctu.
S=1e6*J*SENER/rho; J=current/reference volume for NLGEOM, otherwise 1.
No iterative remodelling. Zero includes the dead zone and formation.
Parameters: Zou et al. (2024), doi:10.1016/j.jmbbm.2024.106463;
s precedent: Tarala et al. (2011), doi:10.1016/j.medengphy.2010.11.005.
"""
import argparse
import os
import numpy as np
from odb_utils import read_mesh, aligned, select_step, signed_volumes

S, B_RATE, RHO0_KG_M3 = 0.35, 1.0, 1760.0
FIELD_NAME = 'RESORP_S035'
DEFAULT_OUTPUT = 'resorption_results'
CHUNK = 50000


def read_case(path, instance_name=None, step_name=None):
    from odbAccess import openOdb
    odb = openOdb(path=path, readOnly=True)
    try:
        names = ([instance_name] if instance_name else
                 [n for n in odb.rootAssembly.instances.keys() if 'CORT' in n.upper()])
        if len(names) != 1:
            raise ValueError('Specify the cortical instance name.')
        instance = odb.rootAssembly.instances[names[0]]
        step = select_step(odb, step_name)
        frame = step.frames[-1]
        mesh = read_mesh(instance)
        energy = aligned(frame.fieldOutputs['SENER'], instance, mesh['labels'])
        if energy.shape != mesh['labels'].shape or not np.isfinite(energy).all() or np.any(energy < 0):
            raise ValueError('SENER must be finite and nonnegative at each cortical element.')
        nlgeom = str(step.nlgeom).strip().upper()
        if nlgeom not in ('ON', 'TRUE', '1', 'OFF', 'FALSE', '0'):
            raise ValueError('Unknown NLGEOM setting.')
        if nlgeom in ('ON', 'TRUE', '1'):
            u = aligned(frame.fieldOutputs['U'], instance, mesh['nodes'], nodal=True)
            if u.shape != mesh['xyz'].shape or not np.isfinite(u).all():
                raise ValueError('Invalid cortical displacements.')
            jacobian = (signed_volumes(mesh['xyz']+u, mesh['index']) /
                        signed_volumes(mesh['xyz'], mesh['index']))
            if np.any(jacobian <= 0) or not np.isfinite(jacobian).all():
                raise ValueError('Invalid or inverted deformed element.')
            energy = energy * jacobian
        return mesh, energy
    finally:
        odb.close()


def resorption_rate(reference_energy, implanted_energy):
    er, ei = np.asarray(reference_energy), np.asarray(implanted_energy)
    if er.shape != ei.shape or er.ndim != 1 or not len(er):
        raise ValueError('Energy arrays must match.')
    if not np.isfinite(er).all() or not np.isfinite(ei).all() or np.any(er < 0) or np.any(ei < 0):
        raise ValueError('Energy values must be finite and nonnegative.')
    rate = np.minimum(B_RATE*1.e6/RHO0_KG_M3 * (ei-(1.-S)*er), 0.)
    if not np.isfinite(rate).all():
        raise ValueError('Invalid resorption rate.')
    return rate


def write_odb(path, mesh, rate):
    from odbAccess import Odb
    from abaqusConstants import THREE_D, DEFORMABLE_BODY, TIME, SCALAR, INTEGRATION_POINT
    odb = Odb(name=os.path.splitext(os.path.basename(path))[0], path=path,
              analysisTitle='Initial bone resorption, s=0.35')
    try:
        part = odb.Part(name='CORTICAL', embeddedSpace=THREE_D, type=DEFORMABLE_BODY)
        for start in range(0, len(mesh['nodes']), CHUNK):
            sl = slice(start, start+CHUNK)
            part.addNodes(labels=tuple(int(n) for n in mesh['nodes'][sl]),
                          coordinates=tuple(tuple(float(x) for x in p) for p in mesh['xyz'][sl]))
        for start in range(0, len(mesh['labels']), CHUNK):
            sl = slice(start, start+CHUNK)
            part.addElements(labels=tuple(int(n) for n in mesh['labels'][sl]),
                connectivity=tuple(tuple(int(n) for n in e) for e in mesh['conn'][sl]), type='C3D4')
        instance = odb.rootAssembly.Instance(name='CORTICAL', object=part)
        step = odb.Step(name='INITIAL_RESORPTION', description='Initial resorption rate',
                        domain=TIME, timePeriod=1.)
        frame = step.Frame(incrementNumber=1, frameValue=1., description='Initial rate; not elapsed time')
        field = frame.FieldOutput(name=FIELD_NAME, type=SCALAR,
            description='Initial resorption rate, s=0.35, kg/m^3/ctu')
        for start in range(0, len(mesh['labels']), CHUNK):
            sl = slice(start, start+CHUNK)
            field.addData(position=INTEGRATION_POINT, instance=instance,
                labels=tuple(int(n) for n in mesh['labels'][sl]),
                data=tuple((float(v),) for v in rate[sl]))
        step.setDefaultField(field)
        odb.save()
    finally:
        odb.close()


def run(reference_path, implant_path, output_dir, reference_instance=None, implant_instance=None, step=None):
    reference_path, implant_path = [os.path.abspath(p.strip().strip('"')) for p in (reference_path, implant_path)]
    if os.path.normcase(reference_path) == os.path.normcase(implant_path):
        raise ValueError('Choose different intact and implanted ODBs.')
    root = os.path.abspath(output_dir.strip().strip('"') or DEFAULT_OUTPUT)
    result = os.path.join(root, os.path.splitext(os.path.basename(implant_path))[0]+'_RESORP_S035.odb')
    if os.path.exists(result):
        raise ValueError('Output already exists; choose another output folder: '+result)
    ref_mesh, er = read_case(reference_path, reference_instance, step)
    imp_mesh, ei = read_case(implant_path, implant_instance, step)
    for key in ('nodes', 'labels', 'conn'):
        if not np.array_equal(ref_mesh[key], imp_mesh[key]):
            raise ValueError('Intact/implanted cortical meshes do not match: '+key)
    if not np.allclose(ref_mesh['xyz'], imp_mesh['xyz'], rtol=0., atol=1.e-4):
        raise ValueError('Intact/implanted reference coordinates do not match.')
    os.makedirs(root, exist_ok=True)
    write_odb(result, imp_mesh, resorption_rate(er, ei))
    print('Saved: '+result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--implant', required=True)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--reference-instance')
    parser.add_argument('--implant-instance')
    parser.add_argument('--step')
    opt = parser.parse_args()
    return run(opt.reference, opt.implant, opt.output, opt.reference_instance, opt.implant_instance, opt.step)


if __name__ == '__main__':
    main()
