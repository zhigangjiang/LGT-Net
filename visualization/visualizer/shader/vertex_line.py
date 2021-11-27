import os
dirname = os.path.dirname(os.path.abspath(__file__))
with open('%s/vertex_line.glsl'%(dirname), 'r') as f:
    src = f.read()

