import os
dirname = os.path.dirname(os.path.abspath(__file__))

with open('%s/fragment_pano.glsl'%(dirname), 'r') as f:
    src = f.read()


