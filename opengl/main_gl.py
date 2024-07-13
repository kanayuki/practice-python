from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def loadOBJ(filename):
    vertices = []
    faces = []

    with open(filename, "r") as file:
        for line in file:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                face = line.split()[1:]
                faces.append([int(vertex.split('/')[0]) - 1 for vertex in face])

    return vertices, faces


# 加载模型
vertices, faces = loadOBJ("your_model.obj")  # 替换为你的OBJ文件路径

# 渲染场景
def drawScene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # glBegin(GL_QUADS)
    # glVertex2f(-0.5, -0.5)
    # glVertex2f(-0.5, 0.5)
    # glVertex2f(0.5, 0.5)
    # glVertex2f(0.5, -0.5)
    # glEnd()

    glTranslatef(0.0,0.0,-5)  # 根据需要调整模型位置
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()
    glutSwapBuffers()

# 初始化OpenGL
def initOpenGL():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(500, 500)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("OpenGL Python Example")
    glutDisplayFunc(drawScene)
    glutIdleFunc(drawScene)
    glClearColor(0.0,0.0,0.0,1.0)
    glutMainLoop()

# 主函数
if __name__ == "__main__":
    initOpenGL()
