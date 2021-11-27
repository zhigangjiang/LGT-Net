import os
import sys
import cv2
import numpy as np
from imageio import imread
import json
import visualization.visualizer.shader as shader
from visualization.visualizer.earcut import earcut
import visualization.visualizer.ArcBall as ArcBall

from PyQt5 import QtWidgets, QtGui, QtOpenGL
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import PyQt5.QtCore as QtCore

import glm
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def GenTexture(img):
    h, w , c = img.shape
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex_id

def GenLayoutVAO(mesh):
    # mesh is N x 2 x 3 x 3
    mesh = mesh.astype(np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, mesh, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    return vao, vbo

def GenLineVAO(lines):
    lines = lines.astype(np.float32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, lines, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    return vao, vbo

def GenPanoShader():
    program = glCreateProgram()
    vertexShader = glCreateShader(GL_VERTEX_SHADER)
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
    
    glShaderSource(vertexShader, shader.vertex_pano.src)
    glShaderSource(fragmentShader, shader.fragment_pano.src)

    glCompileShader(vertexShader)
    glCompileShader(fragmentShader)

    print (glGetShaderInfoLog(vertexShader))
    print (glGetShaderInfoLog(fragmentShader))
    #exit()
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)
    glLinkProgram(program)
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        print(glGetProgramInfoLog(program))
    #print (result)

    um4p = glGetUniformLocation(program, "um4p")
    um4v = glGetUniformLocation(program, "um4v")
    um4m = glGetUniformLocation(program, "um4m")
    ualpha = glGetUniformLocation(program, "alpha")
    uwallNum = glGetUniformLocation(program, "wallNum")
    uwallPoints = glGetUniformLocation(program, "wallPoints")
    #print (program, um4p, um4v, um4m, ualpha)
    
    return program, um4p, um4v, um4m, ualpha, uwallNum, uwallPoints
    
def GenLineShader():
    program = glCreateProgram()
    vertexShader = glCreateShader(GL_VERTEX_SHADER)
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
    geometryShader = glCreateShader(GL_GEOMETRY_SHADER)
    
    glShaderSource(vertexShader, shader.vertex_line.src)
    glShaderSource(fragmentShader, shader.fragment_line.src)
    glShaderSource(geometryShader, shader.geometry_line.src)

    glCompileShader(vertexShader)
    glCompileShader(fragmentShader)
    glCompileShader(geometryShader)

    print (glGetShaderInfoLog(vertexShader))
    print (glGetShaderInfoLog(fragmentShader))
    print (glGetShaderInfoLog(geometryShader))
    #exit()
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)
    glAttachShader(program, geometryShader)
    glLinkProgram(program)
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        print(glGetProgramInfoLog(program))
    #print (result)

    um4p = glGetUniformLocation(program, "um4p")
    um4v = glGetUniformLocation(program, "um4v")
    um4m = glGetUniformLocation(program, "um4m")
    um4f = glGetUniformLocation(program, "um4f")
    return program, um4p, um4v, um4m, um4f



class GLWindow(QOpenGLWidget):
    def __init__(self, img, main, parent=None):
        super(GLWindow, self).__init__(parent)
        self.img = img.copy()
        self.main = main
        self.lastPos = None
        self.width = 720
        self.height = 720
        self.ball = ArcBall.ArcBallT(self.width, self.height)
        self.LastRot = ArcBall.Matrix3fT()
        self.ThisRot = ArcBall.Matrix3fT()
        self.Transform = ArcBall.Matrix4fT()
        self.cam_pos = glm.vec3(0, -6, -2)
        self.cam_tgt = glm.vec3(0, 1, 0)
        self.first_time = True

        glFormat = QtGui.QSurfaceFormat()
        glFormat.setVersion(4, 1)
        glFormat.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        self.setFormat(glFormat)
        QtGui.QSurfaceFormat.setDefaultFormat(glFormat)
        

    def initializeGL(self):
        print (glGetString(GL_VERSION))
        print (glGetString(GL_SHADING_LANGUAGE_VERSION))
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glDepthFunc(GL_LEQUAL)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.texture_id = GenTexture(self.img)
        #self.pano_vao, self.pano_vbo = GenLayoutVAO(self.layout_mesh)
        self.program_id, self.um4p, self.um4v, self.um4m, self.ualpha, self.uwallNum, self.uwallPoints = GenPanoShader()
        self.program_id_line, self.um4p_line, self.um4v_line, self.um4m_line, self.um4f_line = GenLineShader()
        #print (self.program_id, self.um4p, self.um4v, self.um4m)
        self.pano_vao = None
        self.pano_vbo = None

    def updateLayoutMesh(self, wallNum, wallPoints, lines, mesh):
        self.layout_wallNum = wallNum
        self.layout_wallPoints = wallPoints
        self.layout_lines = lines
        self.layout_mesh = mesh
        self.update()
    

    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        self.ball = ArcBall.ArcBallT(self.width, self.height)
        glViewport(0, 0, width, height)
        viewportAspect = float(width) / float(height)
        projection = glm.perspective(90.0/180.0*np.pi, viewportAspect, 0.1, 1000.0);
        view = glm.lookAt(self.cam_pos,    self.cam_tgt,   glm.vec3(0, 0, 1))

        self.mat_proj = np.asarray(projection)
        self.mat_view = np.asarray(view)

    def paintGL(self):
        if self.first_time:
            self.pano_vao, self.pano_vbo = GenLayoutVAO(self.layout_mesh)
            self.line_vao = [GenLayoutVAO(x)[0] for x in self.layout_lines]
            self.first_time = False

        if self.pano_vao is not None:
            glClearColor(1.0, 1.0, 1.0, 1)
            #glClearColor(0.0, 0.0, 0.0, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            glUseProgram(self.program_id)
            glBindVertexArray(self.pano_vao)
            glBindTexture( GL_TEXTURE_2D, self.texture_id)
            
            glUniformMatrix4fv(self.um4p, 1, GL_FALSE, self.mat_proj.astype(np.float32))
            glUniformMatrix4fv(self.um4v, 1, GL_FALSE, self.mat_view.astype(np.float32))
            glUniformMatrix4fv(self.um4m, 1, GL_FALSE, self.Transform.astype(np.float32))

            ### wall ponts update ###
            glUniform1i(self.uwallNum, self.layout_wallNum)
            glUniform2fv(self.uwallPoints, self.layout_wallPoints.shape[0], self.layout_wallPoints.astype(np.float32))
            #print (self.layout_wallPoints)
            #########################
            
            count = self.layout_mesh.reshape([-1, 3]).shape[0]
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            glUniform1f(self.ualpha, 1)
            glDrawArrays(GL_TRIANGLES, 0, count)
            glCullFace(GL_FRONT)
            glUniform1f(self.ualpha, 0.25)
            glDrawArrays(GL_TRIANGLES, 0, count)
            glDisable(GL_CULL_FACE)

            ############## line
            glDisable(GL_DEPTH_TEST)
            glUseProgram(self.program_id_line)
            glBindVertexArray(self.line_vao[0])
            glUniform1i(self.um4f_line, 0)
            glUniformMatrix4fv(self.um4p_line, 1, GL_FALSE, self.mat_proj.astype(np.float32))
            glUniformMatrix4fv(self.um4v_line, 1, GL_FALSE, self.mat_view.astype(np.float32))
            glUniformMatrix4fv(self.um4m_line, 1, GL_FALSE, self.Transform.astype(np.float32))
            glDrawArrays(GL_LINES, 0, self.layout_lines[0].shape[0])
            glBindVertexArray(self.line_vao[1])
            glUniform1i(self.um4f_line, 1)
            glDrawArrays(GL_LINES, 0, self.layout_lines[1].shape[0])
            glBindVertexArray(self.line_vao[2])
            glUniform1i(self.um4f_line, 2)
            glDrawArrays(GL_LINES, 0, self.layout_lines[2].shape[0])
            glEnable(GL_DEPTH_TEST)
            glUseProgram(1)
        glFlush()

    def mousePressEvent(self, event):
        pos = event.pos()
        pt = ArcBall.Point2fT(pos.x(), pos.y())
        self.ball.click(pt)
        
    
    def mouseMoveEvent(self, event):
        pos = event.pos()
        pt = ArcBall.Point2fT(pos.x(), pos.y())
        try:
            ThisQuat = self.ball.drag (pt)
        except:
            return
        ThisRot = ArcBall.Matrix3fSetRotationFromQuat4f (ThisQuat)
        if self.LastRot is None:
            self.LastRot = ArcBall.Matrix3fT()
        if self.Transform is None:
            self.Transform = ArcBall.Matrix4fT()
        self.ThisRot = ArcBall.Matrix3fMulMatrix3f (self.LastRot, ThisRot)
        self.Transform = ArcBall.Matrix4fSetRotationFromMatrix3f(self.Transform, self.ThisRot)

        self.update()
        #print (self.Transform)

    def mouseReleaseEvent(self,event):
        self.LastRot = self.ThisRot 

    def keyPressEvent(self, event):
        step = 0.5
        key = event.key()
        modifiers = event.modifiers()
        if key == ord('S'):
            self.cam_pos[2] -= step
            self.cam_tgt[2] -= step
        elif key == ord('W'):
            self.cam_pos[2] += step
            self.cam_tgt[2] += step
        elif key == ord('A'):
            self.cam_pos[0] -= step
            self.cam_tgt[0] -= step
        elif key == ord('D'):
            self.cam_pos[0] += step
            self.cam_tgt[0] += step

        self.mat_view = np.asarray(glm.lookAt(self.cam_pos, self.cam_tgt, glm.vec3(0, 0, 1)))
        self.update()

    def enterEvent(self, event):
        self.setFocus(True)

    def wheelEvent(self,event):
        numAngle = float(event.angleDelta().y()) / 120
        self.cam_pos[1] += numAngle
        if self.cam_pos[1] > -1: self.cam_pos[1] = -1
        self.mat_view = np.asarray(glm.lookAt(self.cam_pos, self.cam_tgt, glm.vec3(0, 0, 1)))
        self.update()

    
    
        

if __name__ == '__main__':
    img = imread('color.png', pilmode='RGB')
    with open('label.json', 'r') as f:
        label = json.load(f)
    app = QtWidgets.QApplication(sys.argv)
    window = GLWindow(img, label)
    window.show()
    sys.exit(app.exec_())
