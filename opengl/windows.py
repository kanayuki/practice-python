# test.py
import moderngl
import moderngl_window as mglw
import numpy as np


class Test(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernGL Example"
    window_size = (512, 512)
    aspect_ratio = 4 / 3
    resizable = True

    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

        file = open('./opengl/cylinder.shader', 'r')
        code = ''.join(file.readlines())
        self.prog = self.ctx.program(
            vertex_shader="""
        #version 330

        in vec2 in_vert;
        in vec3 in_color;

        uniform float time;

        out vec2 v_vert;
        out vec3 v_color;

        void main() {
            v_vert = in_vert;
            v_color = in_color;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    """,
    fragment_shader=code
    #         fragment_shader="""
    #     #version 330

    #     in vec2 v_vert;
    #     in vec3 v_color;

    #     out vec3 f_color;

    #     void main() {
    #         // Normalized pixel coordinates (from 0 to 1)
    # vec2 uv = v_vert;

    # // Time varying pixel color
    # float iTime=1.0;
    # vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    # // Output to screen
    # // f_color = vec4(col,1.0);
    # f_color = col + v_color*0.01;
    #     }
    # """,
        )

        vertices = np.asarray([
            -0.75, -0.75,  1, 0, 0,
            0.75, -0.75,  0, 1, 0,
            0.0, 0.649,  0, 0, 1
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, self.vbo, "in_vert", "in_color")
 

    def render(self, time, frametime):
        # self.ctx.clear(0.8, 0.5, 0.8, 1.0)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.prog["time"]=time
        # draw(self.ctx)

        self.vao.render()


def draw(ctx):
    fbo = ctx.simple_framebuffer((512, 512))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)

    ctx.renderbuffer()
    np.array().ravel()


Test.run()
