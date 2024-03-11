import moderngl as gl
import struct

ctx = gl.create_context(standalone=True)
program = ctx.program(
    vertex_shader="""
    #version 330 
    out float value;
    out float product;
    
    void main()
    {
        value=gl_VertexID;
        product=gl_VertexID*gl_VertexID;
    }
    """,
    varyings=("value", "product"))

NUM_VERTICES = 10

vao = ctx.vertex_array(program, [])

buffer = ctx.buffer(reserve=NUM_VERTICES * 8)

vao.transform(buffer, vertices=NUM_VERTICES)

data = struct.unpack('20f', buffer.read())

for i in range(0, 20, 2):
    print("value: {}, product: {}".format(*data[i:i + 2]))
