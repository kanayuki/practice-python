from abaqus import *
from abaqusConstants import *
import part, material, section, assembly, step, interaction, load, mesh, job, visualization, xyPlot, displayGroupOdbToolset as dgo, connectorBehavior

# 创建模型数据库和模型
mdb.Model(name='OrigamiModel', modelType=STANDARD_EXPLICIT)

# 创建初始平面几何
s = mdb.models['OrigamiModel'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
s.rectangle(point1=(-50.0, -50.0), point2=(50.0, 50.0))
p = mdb.models['OrigamiModel'].Part(name='Paper', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShell(sketch=s)

# 定义材料和截面属性
mdb.models['OrigamiModel'].Material(name='PaperMaterial')
mdb.models['OrigamiModel'].materials['PaperMaterial'].Elastic(table=((10000.0, 0.3), ))
mdb.models['OrigamiModel'].materials['PaperMaterial'].Density(table=((1e-9, ), ))
mdb.models['OrigamiModel'].HomogeneousShellSection(name='PaperSection', preIntegrate=OFF, material='PaperMaterial', thickness=0.1)
region = p.Set(faces=p.faces, name='PaperRegion')
p.SectionAssignment(region=region, sectionName='PaperSection')
 
# 定义折痕线
a = mdb.models['OrigamiModel'].rootAssembly
a.Instance(name='PaperInstance', part=p, dependent=ON)
e = p.edges
datum_plane = p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=0.0)
p.PartitionFaceByDatumPlane( faces=p.faces,datumPlane=p.datums[datum_plane.id],)

# 划分网格
p.seedPart(size=5.0, deviationFactor=0.1, minSizeFactor=0.1)
p.generateMesh()

# 设置分析步骤
# mdb.models['OrigamiModel'].StaticStep(name='FoldStep', previous='Initial', nlgeom=ON)
mdb.models['OrigamiModel'].ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=1.0 ,
                                     massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 100, 0.0, None, 0, 0,
                                                   0.0, 0.0, 0, None), ),  improvedDtMethod=ON)


# 定义边界条件和施加载荷

# 施加折痕处的位移载荷（例如，将一条边向上折叠）
edge_region = (a.instances['PaperInstance'].edges.findAt((-50.0, 0.0, 0.0) ),)
mdb.models['OrigamiModel'].DisplacementBC(name='FoldBC', createStepName='Step-1', region=edge_region, u3=10.0)

# 施加固定边界条件
fixed_region = (a.instances['PaperInstance'].edges.findAt(((50.0, 0.0, 0.0), )),)
mdb.models['OrigamiModel'].DisplacementBC(name='FixedBC', createStepName='Step-1', region=fixed_region, u1=0.0, u2=0.0, u3=0.0)


# 创建作业并运行
mdb.Job(name='OrigamiJob', model='OrigamiModel', type=ANALYSIS)
# mdb.jobs['OrigamiJob'].submit(consistencyChecking=OFF)
# mdb.jobs['OrigamiJob'].waitForCompletion()

# # 查看结果
# odb = session.openOdb(name='OrigamiJob.odb')
# session.viewports['Viewport: 1'].setValues(displayedObject=odb)
