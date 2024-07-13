from driverUtils import executeOnCaeStartup
from caeModules import *
from abaqus import *
from abaqusConstants import *

vp1 = session.Viewport(name='Viewport: 1', origin=(
    0.0, 0.0),  width=298.449981689453,  height=162.861114501953)
vp1.makeCurrent()
vp1.maximize()

executeOnCaeStartup()
vp1.partDisplay.geometryOptions.setValues(referenceRepresentation=ON)

Mdb()
mymodel = mdb.models['Model-1']

# Create a sketch
sketch1 = mymodel.ConstrainedSketch(name='__profile__',  sheetSize=200.0)
sketch1.Line(point1=(0.0, 0.0), point2=(0.0, 100.0))

# Create a part
part1 = mymodel.Part(name='Part-1', dimensionality=THREE_D,   type=DEFORMABLE_BODY)
part1.BaseWire(sketch=sketch1)


# Create a material
material1 = mymodel.Material(name='Material-1')
material1.Density(table=((1e-09, ), ))
# material1.Elastic(table=((200.0, 0.3), ))
material1.Hyperelastic(
    materialType=ISOTROPIC, testData=OFF, type=MOONEY_RIVLIN,
    volumetricResponse=VOLUMETRIC_DATA, table=((0.2, 6.0, 0.00279), ))

# Create a section
mymodel.TrussSection(name='Section-1', material='Material-1',  area=1.0)

# region1 = part1.Set(cells=part1.cells, name='Set-1')
# part1.SectionAssignment(region=region1, sectionName='Material-1')

e = part1.edges
edges = e.getSequenceFromMask(mask=('[#1 ]', ), )
region = regionToolset.Region(edges=edges)
part1.SectionAssignment(region=region, sectionName='Section-1', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)


# Create a part instance
a = mymodel.rootAssembly
p1 = a.Instance(name='Part-1-1', part=part1, dependent=ON)

# session.viewports['Viewport: 1'].view.fitView()

p2 = a.Instance(name='Part-1-2', part=part1, dependent=ON)
p2.translate(vector=(10.0, 0.0, 0.0))

p3 = a.Instance(name='Part-1-3', part=part1, dependent=ON)
p3.translate(vector=(20.0, 0.0, 0.0))


# Create a step
step1 = mymodel.ExplicitDynamicsStep(name='Step-1', previous='Initial',
                                     timePeriod=3.0, massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 10.0,
                                                                   0.0, None, 0, 0, 0.0, 0.0, 0, None), ), improvedDtMethod=ON)

# step2 = mymodel.StaticStep(name='Step-2', previous='Step-1')
# step3 = mymodel.StaticStep(name='Step-3', previous='Step-2')

# Create a constraint
# mymodel.Coupling(name='Coupling-1', controlPoint=p2.sets['Set-1'],  )


# 接触属性
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
int1.includedPairs.setValuesInStep(
    stepName='Step-1', useAllstar=ON)
int1.contactPropertyAssignments.appendInStep(
    stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))
#: The interaction "Int-1" has been created.


# 删除连接器分配
# del a.sectionAssignments[2]
rp1 = a.ReferencePoint(point=(0.0, 110.0, 0.0))
rp1 = a.referencePoints[rp1.id]
rp2 = a.ReferencePoint(point=(10.0, 110.0, 0.0))
rp2 = a.referencePoints[rp2.id]
rp3 = a.ReferencePoint(point=(20.0, 110.0, 0.0))
rp3 = a.referencePoints[rp3.id]

rp4 = a.ReferencePoint(point=(5.0, 110.0, 0.0))
rp4 = a.referencePoints[rp4.id]
rp5 = a.ReferencePoint(point=(15.0, 110.0, 0.0))
rp5 = a.referencePoints[rp5.id]


set12 = a.Set(referencePoints=(rp1, rp2), name='Set-1-2')
set13 = a.Set(referencePoints=(rp1, rp3), name='Set-1-3')
set23 = a.Set(referencePoints=(rp2, rp3), name='Set-2-3')

# 连接器属性
cs = mymodel.ConnectorSection(name='ConnSect-1', translationalType=JOIN)
# 创建连接器
dtm1 = a.DatumCsysByThreePoints(
    origin=rp1, point1=p1.vertices[1],  coordSysType=CARTESIAN, isZ=True)

wire = a.WirePolyLine(
    points=((rp1, p1.vertices[1]), ), mergeType=IMPRINT,    meshable=False)
a.features.changeKey(fromName=wire.name,  toName='Wire-1')

edges1 = a.edges.getSequenceFromMask(mask=('[#1 ]', ), )
region = a.Set(edges=edges1, name='Wire-1-Set-1')

csa = a.SectionAssignment(sectionName='ConnSect-1', region=region)
a.ConnectorOrientation(region=csa.getSet(), localCsys1=a.datums[dtm1.id])


########################################################################

dtm1 = a.DatumCsysByThreePoints(
    origin=rp2, point1=p2.vertices[1],  coordSysType=CARTESIAN, isZ=True)

wire = a.WirePolyLine(
    points=((rp2, p2.vertices[1]), ), mergeType=IMPRINT,    meshable=False)
a.features.changeKey(fromName=wire.name,  toName='Wire-2')

edges1 = a.edges.getSequenceFromMask(mask=('[#1 ]', ), )
region = a.Set(edges=edges1, name='Wire-2-Set-1')

csa = a.SectionAssignment(sectionName='ConnSect-1', region=region)
a.ConnectorOrientation(region=csa.getSet(), localCsys1=a.datums[dtm1.id])


# 耦合
# refPoints1=(r1[9], r1[22], )
# region2=regionToolset.Region(referencePoints=refPoints1)
mymodel.Coupling(name='Constraint-1',
                 controlPoint=regionToolset.Region(referencePoints=(rp4,)),
                 surface=set12,
                 influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                 alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

# mymodel.Coupling(name='Constraint-2',
#                  controlPoint=regionToolset.Region(referencePoints=rp4),
#                  surface=regionToolset.Region(referencePoints=set12),
#                  influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
#                  alpha=0.0, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)




# Create a load
verts1 = p1.vertices.getSequenceFromMask(mask=('[#1 ]', ), )
verts2 = p2.vertices.getSequenceFromMask(mask=('[#1 ]', ), )
verts3 = p3.vertices.getSequenceFromMask(mask=('[#1 ]', ), )
region = regionToolset.Region(vertices=verts1+verts2+verts3)

# 固定底部的边界条件
mymodel.EncastreBC(name='BC-1', createStepName='Initial',
                   region=region, localCsys=None)



# mymodel.DisplacementBC(name='BC-1', createStepName='Step-1', region=p2.sets['Set-1'],
#     displacement=((0.0, 0.0, 0.0), ))

# mymodel.DisplacementBC(name='BC-2', createStepName='Step-1', region=p3.sets['Set-1'], )


# 旋转半圈
# region 可以使用 Set 或 Tuple 替代
# region = regionToolset.Region(referencePoints=(rp4,))
mymodel.VelocityBC(name='BC-2', createStepName='Step-1',
                   region=(rp4,), v1=0.0, v2=0.0, v3=0.0, vr1=0.0, vr2=3.14*6, vr3=0.0,
                   amplitude=UNSET, localCsys=None, distributionType=UNIFORM, fieldName='')

# Create a mesh
part1.seedPart(size=10.0, deviationFactor=0.1, minSizeFactor=0.1)
part1.generateMesh()

elemType1 = mesh.ElemType(elemCode=T3D2, elemLibrary=EXPLICIT)
edges = part1.edges.getSequenceFromMask(mask=('[#1 ]', ), )
part1.setElementType(regions=(edges, ), elemTypes=(elemType1, ))


# Create a job
a.regenerate()

myjob = mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS,
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                memoryUnits=PERCENTAGE, explicitPrecision=SINGLE,
                nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,
                contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='',
                resultsFormat=ODB, numDomains=1, activateLoadBalancing=False,
                numThreadsPerMpiProcess=1, multiprocessingMode=DEFAULT, numCpus=1)
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


# 调整视图
vp1.odbDisplay.basicOptions.setValues(renderBeamProfiles=ON)

vp1.setValues(displayedObject=a)
vp1.assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
