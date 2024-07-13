# -*- coding: utf-8 -*-

from driverUtils import executeOnCaeStartup
from caeModules import *
from abaqus import *
from abaqusConstants import *

import numpy as np
from scipy.optimize import minimize


def calculate_position(row=5, col=5, space=1):

    y = np.linspace(0.0, w := (row-1) * space, row)-w/2
    x = np.linspace(0.0, h := (col-1) * space, col)-h/2
    return np.meshgrid(x, np.flip(y))


def circle_packing(r, initial_positions):
    num_circles = len(initial_positions)
    # 定义目标函数

    def objective(positions):
        positions = positions.reshape((num_circles, 2))
        R = r + np.max([np.linalg.norm(p) for p in positions])
        # print("R:", R)
        return R

    cons = []
    # 计算小圆之间的重叠约束
    for i in range(num_circles):
        for j in range(i + 1, num_circles):

            def con_fun(positions, i=i, j=j):
                positions = positions.reshape((num_circles, 2))
                dist = np.linalg.norm(positions[i] - positions[j])
                return dist - 2 * r

            cons.append({'type': 'ineq', 'fun': con_fun})

    bounds = [(-5, 5)] * (num_circles * 2)

    # 优化
    result = minimize(objective, initial_positions.flatten(),
                      method='SLSQP', constraints=cons, bounds=bounds)
    print("Final res:", result)

    final_positions = result.x.reshape((num_circles, 2))
    final_R = result.fun
    return final_positions, final_R


def rotation_matrix_from_vectors(u, v):
    # 将向量归一化
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    # 计算旋转轴
    axis = np.cross(u, v)
    axis_norm = np.linalg.norm(axis)

    # 计算旋转角度
    angle = np.arccos(np.dot(u, v))

    # 处理平行和反平行的情况
    if axis_norm == 0:
        if np.dot(u, v) > 0:
            return np.eye(3)  # 两个向量相同，返回单位矩阵
        else:
            return -np.eye(3)  # 两个向量相反，返回负单位矩阵

    axis = axis / axis_norm

    # Rodrigues' 旋转公式
    x, y, z = axis
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    M = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return M


print('enjoy it')
###########################################
# Parameters

row = 2
col = 2
fiber_num = 19
fiber_diameter = 0.5
twist = 0.05

carrier_distance = 30.0
height = 200
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

# Create a part
part1 = mymodel.Part(name='Part-1', dimensionality=THREE_D,
                     type=DEFORMABLE_BODY)


###########################################
# 0 or -1
xd_top = 0 if top_direction == 'Odd' else -1
xd_bottom = 0 if top_direction == 'Even' else -1
yd_left = 0 if left_direction == 'Odd' else -1
yd_right = 0 if left_direction == 'Even' else -1

###########################################

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


# 计算纤维位置
print("Calculating fiber positions...")

np.random.seed(52)
initial_positions = np.random.uniform(-3, 3, (fiber_num, 2))

fiber_positions, yarn_R = circle_packing(
    r=fiber_diameter/2+0.1, initial_positions=initial_positions)

print(f"fiber_positions:\n{fiber_positions}")
print(f"yarn_R: {yarn_R}")
yarn_diameter = yarn_R*2

# 纱线位置
top_x, top_y = calculate_position(row, col, space=yarn_diameter)
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
    # part1.WirePolyLine(points=((yarn_tc, yarn_bc), ), meshable=ON)

    if twist == 0:
        for (x, y) in fiber_positions:
            off = np.array([x, y, 0])
            part1.WirePolyLine(
                points=((yarn_tc+off, yarn_bc+off), ), meshable=ON)
    else:
        # 使用捻度计算模型
        yarn_len = np.linalg.norm(yarn_bc-yarn_tc)
        m = rotation_matrix_from_vectors((0, 0, 1), (yarn_tc-yarn_bc))
        for (x, y) in fiber_positions:
            # off = np.array([x, y, 0])
            # part1.WirePolyLine(points=((yarn_tc+off, yarn_bc+off), ), meshable=ON)

            # theta -> 0 : twist*yarn_en*2*pi : pi/6
            # r -> norm((x,y)), th0 -> arctan2(y,x)
            # p -> {r*cos(theta+th0), r*sin(theta+th0), theta/(2*pi)/twist}

            points = []
            r = np.linalg.norm((x, y))
            th0 = np.arctan2(y, x)
            for theta in np.arange(0, twist*yarn_len*2*pi, pi/6):
                p = (r*np.cos(theta+th0), r*np.sin(theta+th0), theta/(2*pi)/twist)
                points.append((np.dot(m, p) + yarn_bc).tolist())

            # print("points", yarn_len, len(points))
            part1.WireSpline(points=points, meshable=ON)

################################

# Create a material
material1 = mymodel.Material(name='Material-1')
material1.Density(table=((density, ), ))
material1.Elastic(table=((E, PR), ))
# material1.Hyperelastic(
#     materialType=ISOTROPIC, testData=OFF, type=MOONEY_RIVLIN,
#     volumetricResponse=VOLUMETRIC_DATA, table=((1.5, 6.0, 0.000279), ))

area = pi*(fiber_diameter/2)**2
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
    massScaling = ((SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
                   mass_scaling_factor, 0.0, None, 0, 0, 0.0, 0.0, 0, None), )
    s = mymodel.ExplicitDynamicsStep(name=f'Step-{i}', previous=pre,  timePeriod=time_step,
                                     massScaling=massScaling,  improvedDtMethod=ON)

    abaqus_step_list.append(s)

# Intersection

# 建立yarn底部节点的耦合约束，
# 创建yarn底部参考点与carrier参考点的连接器

it = np.nditer(yarn_matrix, flags=['multi_index'])
for val in it:
    if val == 0:
        continue
    mi = it.multi_index

    # 携纱器参考点
    rp_carrier = a.ReferencePoint(point=carriers_coor[mi].tolist())
    rp_carrier = a.referencePoints[rp_carrier.id]

    # 纱线底部参考点
    rp_yarn_bottom = a.ReferencePoint(point=yarns_bottom_coor[mi].tolist())
    rp_yarn_bottom = a.referencePoints[rp_yarn_bottom.id]

    # yarn底部fiber节点
    x, y, z = yarns_bottom_coor[mi].tolist()
    bbox = x-yarn_R, y-yarn_R, z-1, x+yarn_R, y+yarn_R, z+1
    vs = instance_part1.vertices.getByBoundingBox(*bbox)

    # fiber vertices of yarn_{val} , coupling with rp_yarn_bottom
    print('vs', vs)
    fiber_set = a.Set(vertices=vs, name=f'yarn_{val}_node_Set')
    mymodel.Coupling(name=f'Constraint-{val}', controlPoint=(rp_yarn_bottom,),
                     surface=fiber_set, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                     alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    # 坐标系 for connector
    dtm1 = a.DatumCsysByThreePoints(
        origin=rp_carrier, point1=rp_yarn_bottom, coordSysType=CARTESIAN)
    dtm1 = a.datums[dtm1.id]

    wire = a.WirePolyLine(points=((rp_carrier, rp_yarn_bottom), ),
                          mergeType=IMPRINT, meshable=False)
    a.features.changeKey(fromName=wire.name, toName=f'Wire-{val}')

    edges1 = a.edges.findAt((carriers_coor[mi].tolist(),))

    region = a.Set(edges=edges1, name=f'Wire-{val}-Set')
    # 连接器属性
    connSect1 = mymodel.ConnectorSection(
        name='ConnSect-1',  assembledType=TRANSLATOR)
    elastic_0 = connectorBehavior.ConnectorElasticity(components=(1, ),
                                                      behavior=NONLINEAR, table=((24.0, -100.0), (25.0, 100.0)))
    connSect1.setValues(behaviorOptions=(elastic_0, ))

    csa = a.SectionAssignment(sectionName='ConnSect-1', region=region)
    #: The section "ConnSect-1" has been assigned to 1 wire or attachment line.
    a.ConnectorOrientation(region=csa.getSet(), localCsys1=dtm1)

    region = a.Set(referencePoints=(
        rp_carrier, rp_yarn_bottom), name=f'RP-{val}-Set')
    a.engineeringFeatures.PointMassInertia(
        name=f'Inertia-{val}', region=region, mass=1.6e-09, i11=0.01, i22=0.01, i33=0.01,  alpha=0.0, composite=0.0)


#################################################
# yarn顶部fiber节点 耦合约束
top_rp = a.ReferencePoint(point=(0.0, 0.0, height+20.0))
top_rp = a.referencePoints[top_rp.id]

w = row * yarn_diameter
h = col * yarn_diameter
vs = instance_part1.vertices.getByBoundingBox(
    -w, -h, height-1, w, h, height+1)

# print('vs', vs)
top_node_set = a.Set(vertices=vs, name='yarn_top_node_Set')
mymodel.Coupling(name='Constraint-TOP', controlPoint=(top_rp,),
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
