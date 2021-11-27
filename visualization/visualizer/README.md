# 360LayoutVisualizer

This repo is a visualization tool for 360 Manhattan layout based on PyQt5 and OpenGL. The layout format follows <a href='https://github.com/fuenwang/LayoutMP3D'>LayoutMP3D</a>. 
<p align='center'><image src='src/3Dlayout.png' width='100%'></image></p>

First, install the corresponding packages with the following command.
```bash
pip install -r requirements.txt
```
Then, run the script for the visualization of our provided example.
```bash
python visualizer.py --img src/example.jpg  --json src/example.json
```
You can use mouse and keyboard to control the camera.
```yaml
w, a, s, d: translate the camera
left-click: rotate the camera
scroll: zoom in/out
```
<p align='center'><image src='src/demo.png' width='50%'></image></p>
