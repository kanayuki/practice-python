# -*- coding: utf-8 -*-

from driverUtils import executeOnCaeStartup
from caeModules import *
from abaqus import *
from abaqusConstants import *

import numpy as np
 

def calculate_position(row=5, col=5, space=1):

    y = np.linspace(0.0, w := (row-1) * space, row)-w/2
    x = np.linspace(0.0, h := (col-1) * space, col)-h/2
    return np.meshgrid(x, np.flip(y))

# Parameters

print('Starting...')
print(u'天下第一')
# print(u'天下第一')
print('enjoy it')
###########################################
row = 4
col = 4
diameter = 1.0
carrier_distance = 15.0
height = 150
top_direction = 'Even'
left_direction = 'Odd'

cycle = 1
time_step = 0.2

mesh_size = 1.0

density = 1.6e-9
E = 20000.0
PR = 0.3
connector_length = 5.0
mass_scaling_factor = 100.0
first_move_row = True
###########################################
row += 2
col += 2
###########################################
session.journalOptions.setValues(
    replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

vp1 = session.Viewport(name='Viewport: 1', origin=(
    0.0, 0.0),  width=298.449981689453,  height=162.861114501953)
vp1.makeCurrent()
vp1.maximize()

executeOnCaeStartup()
vp1.partDisplay.geometryOptions.setValues(referenceRepresentation=ON)

Mdb()
mymodel = mdb.models['Model-1']
###########################################
# 0 or -1
xd_top = 0 if top_direction == 'Odd' else -1
xd_bottom = 0 if top_direction == 'Even' else -1
yd_left = 0 if left_direction == 'Odd' else -1
yd_right = 0 if left_direction == 'Even' else -1

# sdaaweqw
###########################################
# Create a part
part1 = mymodel.Part(
    name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)

yarn_matrix = np.arange(1, row*col+1).reshape(row, col)
yarn_matrix[xd_top, 0::2] = 0
yarn_matrix[xd_bottom, 1::2] = 0
yarn_matrix[0::2, yd_left] = 0
yarn_matrix[1::2, yd_right] = 0
yarn_matrix[0, 0] = 0
yarn_matrix[0, -1] = 0
yarn_matrix[-1, 0] = 0
yarn_matrix[-1, -1] = 0

print(f"yarn_matrix:\n{yarn_matrix}")

top_x, top_y = calculate_position(row, col, space=diameter*1.5)
bottom_x, bottom_y = calculate_position(row, col, space=carrier_distance)
top_z = np.full((row, col), height)
bottom_z = np.full((row, col), 0)

yarns_top_coor = np.stack((top_x, top_y, top_z), axis=2)
yarns_bottom_coor = np.empty_like(yarns_top_coor)
carriers_coor = np.stack((bottom_x, bottom_y, bottom_z), axis=2)

it = np.nditer(yarn_matrix, flags=['multi_index'])
for val in it:
    if val == 0:
        continue
    mi = it.multi_index
    # print(top_x[mi])
    yarn_tc = yarns_top_coor[mi]
    cc = carriers_coor[mi]

    max_yarn_len = (height**2 + (row*carrier_distance/2) **
           2 + (col*carrier_distance/2)**2)**0.5
    s = (height-connector_length) / max_yarn_len
    yarn_bc = yarn_tc+(cc-yarn_tc)*s
    yarns_bottom_coor[mi] = yarn_bc

    # print("val, yarn_tc, yarn_bc, carrier_c")
    # print(f"carrier_{val}:  {cc}\t yarn_{val}:  {yarn_tc} -> {yarn_bc}\t")

    part1.WirePolyLine(points=((yarn_tc, yarn_bc), ), meshable=ON)

################################

# Create a material
material1 = mymodel.Material(name='Material-1')
material1.Density(table=((density, ), ))
material1.Elastic(table=((E, PR), ))
# material1.Hyperelastic(
#     materialType=ISOTROPIC, testData=OFF, type=MOONEY_RIVLIN,
#     volumetricResponse=VOLUMETRIC_DATA, table=((1.5, 6.0, 0.000279), ))

area = pi*(diameter/2)**2
mymodel.TrussSection(name='Section-1', material='Material-1',  area=area)

# region1 = part1.Set(cells=part1.cells, name='Set-1')
# part1.SectionAssignment(region=region1, sectionName='Material-1')

region = part1.Set(edges=part1.edges, name='yarn-Set-1')
part1.SectionAssignment(region=region, sectionName='Section-1', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)

# Create a part instance
a = mymodel.rootAssembly
instance_part1 = a.Instance(name='Part-1-1', part=part1, dependent=ON)

# Create ABAQUS Step
abaqus_step_list = []

for i in range(1, cycle*4+1):
    pre = 'Initial' if i == 1 else f'Step-{i-1}'

    s = mymodel.ExplicitDynamicsStep(name=f'Step-{i}', previous=pre,  timePeriod=time_step,
                                     massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, mass_scaling_factor, 0.0, None, 0, 0,
                                                   0.0, 0.0, 0, None), ),  improvedDtMethod=ON)

    abaqus_step_list.append(s)

# Intersection

# Nodes
yarn_top_nodes = np.empty_like(yarn_matrix, dtype=object)
yarn_bottom_nodes = np.empty_like(yarn_matrix, dtype=object)
carrier_nodes = np.empty_like(yarn_matrix, dtype=object)

it = np.nditer(yarn_matrix, flags=['multi_index'])
for val in it:
    if val == 0:
        continue
    mi = it.multi_index

    rp = a.ReferencePoint(point=carriers_coor[mi].tolist())
    rp = a.referencePoints[rp.id]

    yarn_top_nodes[mi] = instance_part1.vertices.findAt(
        list(yarns_top_coor[mi]))
    yarn_bottom_nodes[mi] = instance_part1.vertices.findAt(
        yarns_bottom_coor[mi].tolist())
    carrier_nodes[mi] = rp

    # print(f"v{val}", rp, yarns_top_coor[mi], yarn_top_nodes[mi])

    dtm1 = a.DatumCsysByThreePoints(
        origin=rp, point1=yarn_bottom_nodes[mi], coordSysType=CARTESIAN)
    dtm1 = a.datums[dtm1.id]

    wire = a.WirePolyLine(points=((rp, yarn_bottom_nodes[mi]), ),
                          mergeType=IMPRINT, meshable=False)
    a.features.changeKey(fromName=wire.name, toName=f'Wire-{val}')

    edges1 = a.edges.findAt((carriers_coor[mi].tolist(),))

    region = a.Set(edges=edges1, name=f'Wire-{val}-Set-1')

    connSect1 = mymodel.ConnectorSection(
        name='ConnSect-1',  assembledType=TRANSLATOR)
    elastic_0 = connectorBehavior.ConnectorElasticity(components=(1, ),
                                                      behavior=NONLINEAR, table=((24.0, -100.0), (25.0, 100.0)))
    connSect1.setValues(behaviorOptions=(elastic_0, ))

    csa = a.SectionAssignment(sectionName='ConnSect-1', region=region)
    #: The section "ConnSect-1" has been assigned to 1 wire or attachment line.
    a.ConnectorOrientation(region=csa.getSet(), localCsys1=dtm1)

    region = a.Set(vertices=a.vertices.findAt((yarns_bottom_coor[mi].tolist(),)),
                   referencePoints=(rp,), name=f'Set-{val}')
    a.engineeringFeatures.PointMassInertia(
        name=f'Inertia-{val}', region=region, mass=1.6e-09, i11=0.01, i22=0.01, i33=0.01,  alpha=0.0, composite=0.0)

print(111, yarn_top_nodes.flatten().tolist()[1])
print(111, type(yarn_top_nodes.flatten().tolist()[1]))

rp = a.ReferencePoint(point=(0.0, 0.0, height+20.0))
top_rp = a.referencePoints[rp.id]

w = row * diameter
h = col * diameter
vs = instance_part1.vertices.getByBoundingBox(
    -w, -h, height-1, w, h, height+1)

# vs = yarn_top_nodes.ravel()[np.flatnonzero(yarn_top_nodes)].tolist()
print('vs', vs)
top_node_set = a.Set(vertices=vs, name='yarn_top_node_Set-1')
mymodel.Coupling(name='Constraint-1', controlPoint=(top_rp,),
                 surface=top_node_set, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                 alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

# mymodel.MultipointConstraint(name='mpc-1', controlPoint=(top_rp,), surface=top_node_set,
#                              mpcType=BEAM_MPC, userMode=DOF_MODE_MPC, userType=0, csys=None)

#########################################

intProp1 = mymodel.ContactProperty('IntProp-1')
# intProp1=mymodel.interactionProperties['IntProp-1']
intProp1.TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
        0.1, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION,
    fraction=0.005, elasticSlipStiffness=None)
intProp1.NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON,
    constraintEnforcementMethod=DEFAULT)
#: The interaction property "IntProp-1" has been created.

int1 = mymodel.ContactExp(name='Int-1', createStepName='Step-1')
int1.includedPairs.setValuesInStep(stepName='Step-1', useAllstar=ON)
int1.contactPropertyAssignments.appendInStep(
    stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))
#: The interaction "Int-1" has been created.

##########################################################

# Create a load
# 创建幅值
amp1 = mymodel.TabularAmplitude(
    name='Amp-1', timeSpan=STEP,  smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (time_step, 1.0)))

# mymodel.EncastreBC(name='BC-fix-top-node', createStepName='Initial',
#                    region=(top_rp,), localCsys=None)
mymodel.DisplacementBC(name=f'BC-top-move', createStepName='Step-1', region=(top_rp,), u1=0, u2=0, u3=5/0.8, ur1=0, ur2=0, ur3=0,
                       amplitude="Amp-1", distributionType=UNIFORM, fieldName='', localCsys=None)

it = np.nditer(yarn_matrix, flags=['multi_index'])
for val in it:
    if val == 0:
        continue
    mi = it.multi_index
    point = carriers_coor[mi].tolist()

    # print(f"CS yarn_{val} at {point} reference to carrier")
    region = (a.referencePoints.findAt(point),)

    mymodel.DisplacementBC(name=f'BC-{val}', createStepName='Initial', region=region, u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET,
                           amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)

zero_matrix = np.zeros((row, col))
one_matrix = np.ones((row, col))

u1s1 = zero_matrix
u2s1 = one_matrix.copy()
u2s1[:, 1::2] = -1
u2s1[:, [0, -1]] = 0
if top_direction == 'Even':
    u2s1 *= -1

u1s2 = one_matrix.copy()
u1s2[0::2, :] = -1
u1s2[[0, -1], :] = 0
u2s2 = zero_matrix
if left_direction == 'Even':
    u1s2 *= -1

u1s3 = zero_matrix
u2s3 = -u2s1

u1s4 = -u1s2
u2s4 = zero_matrix

if first_move_row:
    u1s1, u1s2 = u1s2, u1s1
    u2s1, u2s2 = u2s2, u2s1

    u1s3, u1s4 = u1s4, u1s3
    u2s3, u2s4 = u2s4, u2s3

print(f"u1s1: \n{u1s1}")
print(f"u2s1: \n{u2s1}")
print(f"u1s2: \n{u1s2}")    
print(f"u2s2: \n{u2s2}")
print(f"u1s3: \n{u1s3}")
print(f"u2s3: \n{u2s3}")
print(f"u1s4: \n{u1s4}")
print(f"u2s4: \n{u2s4}")

print(f"initial:\n {yarn_matrix}")

for i, abaqus_step in enumerate(abaqus_step_list):
    it = np.nditer(yarn_matrix, flags=['multi_index'])
    for val in it:
        if val == 0:
            continue
        mi = it.multi_index

        match i % 4+1:
            case 1:
                u1 = u1s1[mi] * carrier_distance
                u2 = u2s1[mi] * carrier_distance
            case 2:
                u1 = u1s2[mi] * carrier_distance
                u2 = u2s2[mi] * carrier_distance
            case 3:
                u1 = u1s3[mi] * carrier_distance
                u2 = u2s3[mi] * carrier_distance
            case 4:
                u1 = u1s4[mi] * carrier_distance
                u2 = u2s4[mi] * carrier_distance

        # print(f"CS\t BC-{val} \t {abaqus_step.name} : {x, y} -> {u1, u2} ")
        mymodel.boundaryConditions[f'BC-{val}'].setValuesInStep(
            stepName=abaqus_step.name, u1=u1, u2=u2, amplitude='Amp-1')


    ym = yarn_matrix
    xds = 1 if left_direction == 'Odd' else -1
    yds = 1 if top_direction == 'Odd' else -1
    if first_move_row:

        match i % 4+1:
            case 1:
                ym[1:-1:2, :] = np.roll(ym[1:-1:2, :], xds, axis=1)
                ym[2:-1:2, :] = np.roll(ym[2:-1:2, :], -xds, axis=1)
            case 2:
                ym[:, 1:-1:2] = np.roll(ym[:, 1:-1:2], yds, axis=0)
                ym[:, 2:-1:2] = np.roll(ym[:, 2:-1:2], -yds, axis=0)
            case 3:
                ym[1:-1:2, :] = np.roll(ym[1:-1:2, :], -xds, axis=1)
                ym[2:-1:2, :] = np.roll(ym[2:-1:2, :], xds, axis=1)
            case 4:
                ym[:, 1:-1:2] = np.roll(ym[:, 1:-1:2], -yds, axis=0)
                ym[:, 2:-1:2] = np.roll(ym[:, 2:-1:2], yds, axis=0)

    else:
        match i % 4+1:
            case 1:
                ym[:, 1:-1:2] = np.roll(ym[:, 1:-1:2], yds, axis=0)
                ym[:, 2:-1:2] = np.roll(ym[:, 2:-1:2], -yds, axis=0)
            case 2:
                ym[1:-1:2, :] = np.roll(ym[1:-1:2, :], xds, axis=1)
                ym[2:-1:2, :] = np.roll(ym[2:-1:2, :], -xds, axis=1)
            case 3:
                ym[:, 1:-1:2] = np.roll(ym[:, 1:-1:2], -yds, axis=0)
                ym[:, 2:-1:2] = np.roll(ym[:, 2:-1:2], yds, axis=0)
            case 4:
                ym[1:-1:2, :] = np.roll(ym[1:-1:2, :], -xds, axis=1)
                ym[2:-1:2, :] = np.roll(ym[2:-1:2, :], xds, axis=1)
    yarn_matrix = ym
    print(f"{abaqus_step.name}:\n {yarn_matrix}")

# Create a mesh
part1.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
part1.generateMesh()

elemType1 = mesh.ElemType(elemCode=T3D2, elemLibrary=EXPLICIT)
part1.setElementType(regions=(part1.edges, ), elemTypes=(elemType1, ))

# Create a job
a.regenerate()

myjob = mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS,
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                memoryUnits=PERCENTAGE, explicitPrecision=SINGLE,
                nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,
                contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='',
                resultsFormat=ODB, numDomains=2, activateLoadBalancing=False,
                numThreadsPerMpiProcess=1, multiprocessingMode=DEFAULT, numCpus=2)
# myjob.submit()
# myjob.waitForCompletion()

# Create a history output request
# myjob.writeInput()

# Create a field output request
# myodb = session.openOdb(name='Job-1.odb')
# vp1.setValues(displayedObject=myodb)
# vp1.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))

# myframe = myodb.steps['Step-1'].frames[-1]
# myframe.fieldOutputs['S'].setValuesInStep(step=0,
#                                         region=p2.sets['Set-1'],
#                                     position=NODAL)
# myframe.fieldOutputs['S'].setValuesInStep(step=0,
# )
# myodb.close()

vp1.odbDisplay.basicOptions.setValues(renderBeamProfiles=ON,
                                      connectorDisplay=ON, highlightConnectorPts=ON, showConnectorAxes=OFF,
                                      showConnectorType=OFF)

vp1.setValues(displayedObject=a)
vp1.assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)

vp1.view.setValues(width=125.234, height=43.749,
                   cameraPosition=(-0.524496, -0.151751, 581.586), cameraUpVector=(0, 1, 0),
                   cameraTarget=(-0.524496, -0.151751, 110.15), viewOffsetX=0, viewOffsetY=0)
vp1.view.setValues(session.views['Front'])
