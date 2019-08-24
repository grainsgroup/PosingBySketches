"""
Project: PosingBySketching
Version: 4.0

==========================
OpenVR Compatible (HTC Vive)
==========================

OpenVR Compatible head mounted display
It uses a python wrapper to connect with the SDK
"""
import threading
import openvr
import bpy
import math
import sys
import copy
from enum import Enum
from mathutils import Quaternion
from mathutils import Matrix
from mathutils import Vector
import winsound

import time

from . import HMD_Base

from ..lib import (
        checkModule,
        )

import datetime
from scipy.optimize import linear_sum_assignment
import numpy as np

currObject = ""
currBone = ""
currObject_l = ""
currBone_l = ""


# Algorithm from "New Algorithms for 2D and 3D Point Matching: Pose Estimation and Correpondance" used also in
#   1) A New Point Matching Algorithm for Non-Rigid Registration
#   2) Enhancing Character Posing by a Sketch-Based Interaction
class softAss_detAnnealing_3(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    # Compute 3D distance
    def compute_dist(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))

    # Move the first bone at the beggining og the stroke
    def set_init_cond(self):
        bones = bpy.data.objects['Armature'].pose.bones  # <H
        stroke = bpy.data.curves['Stroke'].splines[0]  # <H

        # TODO: Find root bones
        for b in bones:
            if b.parent == None:
                print(bpy.data.objects['Armature'].matrix_world * b.head)

                pmatrix = b.bone.matrix_local
                omatrix = bpy.data.objects['Armature'].matrix_world


                target_loc = bpy.data.objects['StrokeObj'].matrix_world * stroke.bezier_points[0].co
                b.location = omatrix.inverted() * pmatrix.inverted() * target_loc

                bpy.context.scene.update()
                print(bpy.data.objects['Armature'].matrix_world * b.head)

    def run(self):
        bones = bpy.data.objects['Armature'].pose.bones  # <H
        # stroke = bpy.data.curves['Stroke'].splines[0] # <H

        dict = {}
        stroke_points = []
        num_points = 0
        for k in range(0, len(bpy.data.curves['Stroke'].splines)):
            stroke = bpy.data.curves['Stroke'].splines[k]
            for j in range(0, len(stroke.bezier_points)):
                p = stroke.bezier_points[j]
                # World position of the point
                point = bpy.data.objects['StrokeObj'].matrix_world * p.co
                # Add into the dictionoary for correspondance
                dict[num_points] = [k, j]  # [spline_index, point_index]
                # Add in the point list
                stroke_points.append(point)
                num_points += 1

        # [DEBUG]:
        # print(dict)
        # for i in range(0, num_points):
        #    print(dict[i])
        #    print(dict[i][0], dict[i][1])

        # INIT PARAMETERS
        # values of the paper
        # beta_f = 0.2
        # beta_r = 1.075
        # beta = 0.00091
        # alpha = 0.03
        # I0 = 4
        # I1 = 30

        # custom value
        beta_f = 0.2
        beta_r = 1.9
        beta = 0.00091
        alpha = 0.03
        I0 = 4
        I1 = 30

        iteration = 0

        # set starting condition
        self.set_init_cond()

        while beta <= beta_f:
            print('iteration:', iteration)

            Qjk = []

            # Compute Qjk
            for k in range(0, num_points):
                p = bpy.data.curves['Stroke'].splines[dict[k][0]].bezier_points[dict[k][1]]

                # World position of the point
                point = bpy.data.objects['StrokeObj'].matrix_world * p.co

                costs_row = []
                for i in range(0, len(bones)):
                    b = bones[i]
                    # World position of the bone
                    tail = bpy.data.objects['Armature'].matrix_world * b.tail  # <H
                    head = bpy.data.objects['Armature'].matrix_world * b.head  # <H
                    # center = bpy.data.objects['Armature'].matrix_world * b.center # <H

                    dist = self.compute_dist(tail, point)
                    # length_diff = abs(b.length - compute_dist(point,head))
                    # costs_row.append(dist + length_diff)
                    costs_row.append(-(dist - alpha))

                Qjk.append(costs_row)

            m0 = np.asarray(Qjk)

            # Deterministic annealing
            for i in range(0, num_points):
                for j in range(0, len(bones)):
                    m0[i, j] = math.exp(beta * m0[i, j])

                    # TODO: Add outlier row and cols

            # Sinkhorn's method DO until m converges
            m1 = np.ones((num_points, len(bones)))

            for i in range(0, I1):  # TODO: set coverage threshold
                for i in range(0, num_points):
                    for j in range(0, len(bones)):
                        m1[i, j] = m0[i, j] / np.sum(m0[i])

                for i in range(0, num_points):
                    for j in range(0, len(bones)):
                        m0[i, j] = m1[i, j] / np.sum(m1[:, j])

                        # [DEBUG]: Rows - Cols sum up
            # print("rows - cols sum up")
            # for i in range (0,num_points):
            #    print (np.sum(m0[i]))

            # for j in range (0, len(bones)):
            #    print(np.sum(m0[:,j]))

            # Softassign - ENERGY FORMULATION E3D
            E3D = []
            # Compute E3D
            for k in range(0, num_points):
                p = bpy.data.curves['Stroke'].splines[dict[k][0]].bezier_points[dict[k][1]]

                # World position of the point
                point = bpy.data.objects['StrokeObj'].matrix_world * p.co

                costs_row = []
                for i in range(0, len(bones)):
                    b = bones[i]
                    # World position of the bone
                    tail = bpy.data.objects['Armature'].matrix_world * b.tail  # <H
                    head = bpy.data.objects['Armature'].matrix_world * b.head  # <H
                    # center = bpy.data.objects['Armature'].matrix_world * b.center # <H

                    dist = self.compute_dist(tail, point)
                    # length_diff = abs(b.length - compute_dist(point,head))
                    # costs_row.append(dist + length_diff)
                    costs_row.append(dist - alpha * m0[k, i])

                E3D.append(costs_row)

            cost = np.transpose(m0) * np.transpose(E3D)
            row_ind, col_ind = linear_sum_assignment(cost)
            print(col_ind)
            print(cost[row_ind, col_ind].sum())

            # Update pose parameters
            for i in range(0, len(bones)):
                b = bones[i]
                # Set target positions
                #
                bpy.data.objects[b.name].location = copy.deepcopy(bpy.data.objects['StrokeObj'].matrix_world *
                                                                  bpy.data.curves['Stroke'].splines[
                                                                      dict[col_ind[i]][0]].bezier_points[
                                                                      dict[col_ind[i]][1]].co)

                # Articulate armature
                constr = b.constraints['Damped Track']
                constr.target = bpy.data.objects[b.name]

            iteration += 1
            beta = beta * beta_r




class State(Enum):
    IDLE = 1
    DECISIONAL = 2
    INTERACTION_LOCAL = 3
    NAVIGATION_ENTER = 4
    NAVIGATION = 5
    NAVIGATION_EXIT = 6
    ZOOM_IN = 7
    ZOOM_OUT = 8
    CAMERA_MOVE_CONT = 9
    CAMERA_ROT_CONT = 10
    SCALING = 11
    CHANGE_AXES = 12
    DRAWING = 13
    TRACKPAD_BUTTON_DOWN = 14

class StateLeft(Enum):
    IDLE = 1
    DECISIONAL = 2
    INTERACTION_LOCAL = 3
    NAVIGATION = 5
    SCALING = 11
    CHANGE_AXES = 12

class OpenVR(HMD_Base):
    ctrl_index_r = 0
    ctrl_index_l = 0
    tracker_index = 0
    hmd_index = 0
    curr_axes_r = 0
    curr_axes_l = 0
    state = State.IDLE
    state_l = StateLeft.IDLE

    diff_rot = Quaternion()
    diff_loc = bpy.data.objects['Controller.R'].location
    initial_loc = Vector((0,0,0))
    initial_rot = Quaternion()

    diff_rot_l = Quaternion()
    diff_loc_l = bpy.data.objects['Controller.L'].location
    initial_loc_l = Vector((0, 0, 0))
    initial_rot_l = Quaternion()

    diff_distance = 0
    initial_scale = 0
    trans_matrix = bpy.data.objects['Camera'].matrix_world * bpy.data.objects['Origin'].matrix_world
    diff_trans_matrix = bpy.data.objects['Camera'].matrix_world * bpy.data.objects['Origin'].matrix_world

    objToControll = ""
    boneToControll = ""
    objToControll_l = ""
    boneToControll_l = ""
    zoom = 1
    rotFlag = True
    axes = ['LOC/ROT_XYZ','LOC_XYZ','LOC_X','LOC_Y','LOC_Z','ROT_XYZ','ROT_X','ROT_Y','ROT_Z']

    gui_obj = ['Camera', 'Origin',
               'Controller.R', 'Controller.L',
               'Text.R', 'Text.L']


    def __init__(self, context, error_callback):
        super(OpenVR, self).__init__('OpenVR', True, context, error_callback)
        checkModule('hmd_sdk_bridge')

    def _getHMDClass(self):
        """
        This is the python interface to the DLL file in hmd_sdk_bridge.
        """
        from bridge.hmd.openvr import HMD
        return HMD

    @property
    def projection_matrix(self):
        if self._current_eye:
            matrix = self._hmd.getProjectionMatrixRight(self._near, self._far)
        else:
            matrix = self._hmd.getProjectionMatrixLeft(self._near, self._far)

        self.projection_matrix = matrix
        return super(OpenVR, self).projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._projection_matrix[self._current_eye] = \
            self._convertMatrixTo4x4(value)

    def init(self, context):
        """
        Initialize device

        :return: return True if the device was properly initialized
        :rtype: bool
        """

        vrSys = openvr.init(openvr.VRApplication_Scene)
        self.ctrl_index_r, self.ctrl_index_l, self.tracker_index, self.hmd_index = self.findControllers(vrSys)
        if bpy.data.objects.get('StrokeObj') is None:
            self.create_curve()
        bpy.data.window_managers['WinMan'].virtual_reality.lock_camera = True

        try:
            HMD = self._getHMDClass()
            self._hmd = HMD()

            # bail out early if we didn't initialize properly
            if self._hmd.get_state_bool() == False:
                raise Exception(self._hmd.get_status())

            # Tell the user our status at this point.
            self.status = "HMD Init OK. Make sure lighthouses running else no display."

            # gather arguments from HMD
            self.setEye(0)
            self.width = self._hmd.width_left
            self.height = self._hmd.height_left

            self.setEye(1)
            self.width = self._hmd.width_right
            self.height = self._hmd.height_right

            # initialize FBO
            if not super(OpenVR, self).init():
                raise Exception("Failed to initialize HMD")

            # send it back to HMD
            if not self._setup():
                raise Exception("Failed to setup OpenVR Compatible HMD")

        except Exception as E:
            self.error("OpenVR.init", E, True)
            self._hmd = None
            return False

        else:
            return True

    def _setup(self):
        return self._hmd.setup(self._color_texture[0], self._color_texture[1])

    # ---------------------------------------- #
    # Functions
    # ---------------------------------------- #
    ## Find the index of the two controllers
    def findControllers(self, vrSys):
        r_index, l_index, tracker_index, hmd_index = -1, -1, -1, -1

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_Invalid:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - ")
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_HMD:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - HMD")
                hmd_index = i
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_TrackingReference:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - TrackingReference")
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_Controller:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - Controller")
                if r_index == -1:
                    r_index = i
                else:
                    l_index = i
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == 3:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - VIVE Tracker")
                tracker_index = i

        print('r_index = ', r_index, ' l_index = ', l_index)
        return r_index, l_index, tracker_index, hmd_index

    def setController(self):
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        poses = poses_t()
        openvr.VRCompositor().waitGetPoses(poses, len(poses), None, 0)

        matrix = poses[self.ctrl_index_r].mDeviceToAbsoluteTracking
        matrix2 = poses[self.ctrl_index_l].mDeviceToAbsoluteTracking

        try:
            camera = bpy.data.objects["Camera"]
            ctrl = bpy.data.objects["Controller.R"]
            ctrl_l = bpy.data.objects["Controller.L"]


            self.trans_matrix = camera.matrix_world * bpy.data.objects['Origin'].matrix_world
            RTS_matrix = Matrix(((matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]),
                                 (matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]),
                                 (matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]),
                                 (0, 0, 0, 1)))

            RTS_matrix2 = Matrix(((matrix2[0][0], matrix2[0][1], matrix2[0][2], matrix2[0][3]),
                                 (matrix2[1][0], matrix2[1][1], matrix2[1][2], matrix2[1][3]),
                                 (matrix2[2][0], matrix2[2][1], matrix2[2][2], matrix2[2][3]),
                                 (0, 0, 0, 1)))

            # Interaction state active
            if(self.rotFlag):
                ctrl.matrix_world = self.trans_matrix * RTS_matrix
                bpy.data.objects["Text.R"].location = ctrl.location
                bpy.data.objects["Text.R"].rotation_quaternion = ctrl.rotation_quaternion * Quaternion((0.707, -0.707, 0, 0))

                ctrl_l.matrix_world = self.trans_matrix * RTS_matrix2
                bpy.data.objects["Text.L"].location = ctrl_l.location
                bpy.data.objects["Text.L"].rotation_quaternion = ctrl_l.rotation_quaternion * Quaternion((0.707, -0.707, 0, 0))

            # Navigation state active
            else:
                diff_rot_matr = self.diff_rot.to_matrix()
                inverted_matrix = RTS_matrix * diff_rot_matr.to_4x4()
                inverted_matrix = inverted_matrix.inverted()
                stMatrix = self.diff_trans_matrix * inverted_matrix
                quat = stMatrix.to_quaternion()
                camera.rotation_quaternion = quat



        except:
            print("ERROR: ")

    def changeSelection(self,obj,bone,selectState):

        if selectState:
            print("SELECT: ", obj, bone)
            if obj != "":
                if bone != "":
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.ops.object.mode_set(mode='POSE')
                    bpy.data.objects[obj].data.bones[bone].select = True

                else:
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.ops.object.mode_set(mode='OBJECT')

        else:
            print ("DESELECT: ", obj, bone)
            if obj != "":
                if bone != "":
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.ops.object.mode_set(mode='POSE')
                    bpy.data.objects[obj].data.bones[bone].select = False
                    bpy.data.objects[obj].select = False
                else:
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.data.objects[obj].select = False
                    bpy.ops.object.mode_set(mode='OBJECT')

    ## Computes distance from controller
    def computeTargetObjDistance(self, Object, Bone, isRotFlag):

        if isRotFlag:
            tui = bpy.data.objects['Controller.R']
        else:
            tui = bpy.data.objects['Controller.L']

        obj = bpy.data.objects[Object]
        if Bone != "":
            pbone = obj.pose.bones[Bone]
            return (math.sqrt(pow((tui.location[0] - (pbone.center[0] + obj.location[0])), 2) + pow(
                (tui.location[1] - (pbone.center[1] + obj.location[1])), 2) + pow(
                (tui.location[2] - (pbone.center[2] + obj.location[2])), 2)))
        else:
            loc = obj.matrix_world.to_translation()
            return (math.sqrt(pow((tui.location[0] - loc[0]), 2) + pow((tui.location[1] - loc[1]), 2) + pow(
                (tui.location[2] - loc[2]), 2)))

    ## Returns the object closest to the Controller
    def getClosestItem(self, isRight):
        dist = sys.float_info.max
        cObj = ""
        cBone = ""
        distThreshold = 0.5

        for object in bpy.data.objects:
            if object.type == 'ARMATURE':
                if not ':TEST_REF' in object.name:
                    for bone in object.pose.bones:
                        currDist = self.computeTargetObjDistance(object.name, bone.name, isRight)
                        bone.bone_group = None
                        if (currDist < dist and currDist < distThreshold):
                            dist = currDist
                            cObj = object.name
                            cBone = bone.name


            else:
                #if object.type != 'CAMERA' and not object.name in self.gui_obj:
                if not object.name in self.gui_obj:
                    currDist = self.computeTargetObjDistance(object.name, "", isRight)
                    # print(object.name, bone.name, currDist)
                    if (currDist < dist and currDist < distThreshold):
                        dist = currDist
                        cObj = object.name
                        cBone = ""



        # Select the new closest item
        print(cObj, cBone)
        print("--------------------------------")
        if(cBone!=""):
            bpy.data.objects[cObj].pose.bones[cBone].rotation_mode = 'QUATERNION'
            #bpy.data.objects[cObj].pose.bones[cBone].bone_group = bpy.data.objects[cObj].pose.bone_groups["SelectedBones"]

        return cObj, cBone

    ## Resets the original transformation when constraints movement are used
    def applyConstraint(self, isRight):
        if isRight:
            type = self.axes[self.curr_axes_r].split('_')[0]
            axes = self.axes[self.curr_axes_r].split('_')[1]
            obj = self.objToControll
            bone = self.boneToControll
            init_loc = self.initial_loc
            init_rot = self.initial_rot

        else:
            type = self.axes[self.curr_axes_l].split('_')[0]
            axes = self.axes[self.curr_axes_l].split('_')[1]
            obj = self.objToControll_l
            bone = self.boneToControll_l
            init_loc = self.initial_loc_l
            init_rot = self.initial_rot_l

        if type == 'LOC':
            if bone!="":
                bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                bpy.data.objects[obj].pose.bones[bone].rotation_euler = init_rot
                bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'

            else:
                bpy.data.objects[obj].rotation_mode = 'XYZ'
                bpy.data.objects[obj].rotation_euler = init_rot
                bpy.data.objects[obj].rotation_mode = 'QUATERNION'

            if axes == 'X':
                if bone != "":
                    bpy.data.objects[obj].pose.bones[bone].location[1] = init_loc[1]
                    bpy.data.objects[obj].pose.bones[bone].location[2] = init_loc[2]
                else:
                    bpy.data.objects[obj].location[1] = init_loc[1]
                    bpy.data.objects[obj].location[2] = init_loc[2]

            if axes == 'Y':
                if bone != "":
                    bpy.data.objects[obj].pose.bones[bone].location[0] = init_loc[0]
                    bpy.data.objects[obj].pose.bones[bone].location[2] = init_loc[2]
                else:
                    bpy.data.objects[obj].location[0] = init_loc[0]
                    bpy.data.objects[obj].location[2] = init_loc[2]

            if axes == 'Z':
                if bone != "":
                    bpy.data.objects[obj].pose.bones[bone].location[0] = init_loc[0]
                    bpy.data.objects[obj].pose.bones[bone].location[1] = init_loc[1]
                else:
                    bpy.data.objects[obj].location[0] = init_loc[0]
                    bpy.data.objects[obj].location[1] = init_loc[1]

        if type == 'ROT':
            if bone!="":
                bpy.data.objects[obj].pose.bones[bone].location = init_loc
            else:
                bpy.data.objects[obj].location = init_loc

            if axes == 'X':
                if bone!="":
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'
                else:
                    bpy.data.objects[obj].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].rotation_mode = 'QUATERNION'

            if axes == 'Y':
                if bone!="":
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'
                else:
                    bpy.data.objects[obj].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].rotation_mode = 'QUATERNION'

            if axes == 'Z':
                if bone!="":
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'
                else:
                    bpy.data.objects[obj].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].rotation_mode = 'QUATERNION'

    ## Create a curve from a set of points
    def create_curve(self):
        name = "Stroke"
        curvedata = bpy.data.curves.new(name=name, type='CURVE')
        curvedata.dimensions = '3D'
        curvedata.fill_mode = 'FULL'
        curvedata.bevel_depth = 0.01

        ob = bpy.data.objects.new(name + "Obj", curvedata)
        bpy.context.scene.objects.link(ob)
        ob.show_x_ray = True

    def add_spline(self, point):
        curvedata = bpy.data.curves['Stroke']
        polyline = curvedata.splines.new('BEZIER')
        polyline.resolution_u = 1
        polyline.bezier_points[0].co = point

    ## Add new point to the curve
    def update_curve(self, point):
        polyline = bpy.data.curves['Stroke'].splines[-1]
        polyline.bezier_points.add(1)
        polyline.bezier_points[-1].co = point
        polyline.bezier_points[-1].handle_left = point
        polyline.bezier_points[-1].handle_right = point
        print (datetime.datetime.now())

    def remove_spline(self):
        polyline = bpy.data.curves['Stroke'].splines[-1]
        bpy.data.curves['Stroke'].splines.remove(polyline)











    # ---------------------------------------- #
    # Main Loop
    # ---------------------------------------- #
    def loop(self, context):
        """
        Get fresh tracking data
        """
        try:
            data = self._hmd.update()
            self._eye_orientation_raw[0] = data[0]
            self._eye_orientation_raw[1] = data[2]
            self._eye_position_raw[0] = data[1]
            self._eye_position_raw[1] = data[3]


            self.setController()

            # ctrl_state contains the value of the button
            idx, ctrl_state = openvr.IVRSystem().getControllerState(self.ctrl_index_r)
            idx_l, ctrl_state_l = openvr.IVRSystem().getControllerState(self.ctrl_index_l)

            ctrl = bpy.data.objects['Controller.R']
            ctrl_l = bpy.data.objects['Controller.L']
            camera = bpy.data.objects['Camera']

            ########## Right_Controller_States ##########

            if self.state == State.IDLE:
                bpy.data.objects["Text.R"].data.body = "Idle\n" + self.objToControll + "-" + self.boneToControll

                # DECISIONAL
                if (ctrl_state.ulButtonPressed == 4):
                    print("IDLE -> DECISIONAL")
                    self.changeSelection(self.objToControll, self.boneToControll, False)
                    self.state = State.DECISIONAL

                # # INTERACTION_LOCAL - VR_BLENDER
                # if ctrl_state.ulButtonPressed == 8589934592 and self.objToControll != "":
                #     print("IDLE -> INTERACTION LOCAL")
                #     self.state = State.INTERACTION_LOCAL
                #     self.curr_axes_r = 0
                #
                #     if self.boneToControll != "":
                #         self.diff_rot = ctrl.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].matrix.to_quaternion()
                #         self.diff_loc = bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].matrix.to_translation() - ctrl.location
                #         self.initial_loc = copy.deepcopy(bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].location)
                #         bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].rotation_mode = 'XYZ'
                #         self.initial_rot = copy.deepcopy(bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].rotation_euler)
                #         bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].rotation_mode = 'QUATERNION'
                #
                #     else:
                #         self.diff_rot = ctrl.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll].rotation_quaternion
                #         self.diff_loc = bpy.data.objects[self.objToControll].location - ctrl.location
                #         self.initial_loc = copy.deepcopy(bpy.data.objects[self.objToControll].location)
                #         bpy.data.objects[self.objToControll].rotation_mode = 'XYZ'
                #         self.initial_rot = copy.deepcopy(bpy.data.objects[self.objToControll].rotation_euler)
                #         bpy.data.objects[self.objToControll].rotation_mode = 'QUATERNION'

                # DRAWING SKETCHES - POSING_SKETCHES
                if ctrl_state.ulButtonPressed == 8589934592:
                    print("IDLE -> DRAWING")
                    self.add_spline(bpy.data.objects['Controller.R'].location)
                    self.state = State.DRAWING

                #TRACKPAD.R
                if ctrl_state.ulButtonPressed == 4294967296:
                    self.state = State.TRACKPAD_BUTTON_DOWN

                # NAVIGATION
                if ctrl_state.ulButtonPressed == 2:
                    print("IDLE -> NAVIGATION")
                    self.state = State.NAVIGATION_ENTER

            elif self.state == State.DRAWING:
                self.update_curve(bpy.data.objects['Controller.R'].location)

                if (ctrl_state.ulButtonPressed != 8589934592):
                    self.state = State.IDLE

            elif self.state == State.TRACKPAD_BUTTON_DOWN:

                if ctrl_state.ulButtonPressed != 4294967296:
                    x, y = ctrl_state.rAxis[0].x, ctrl_state.rAxis[0].y
                    # Apply rotation for X setup otherwiise + setup
                    x1, y1 = x * 0.707 - y * -0.707, x * -0.707 + y * 0.707
                    x, y = x1, y1

                    if x > 0 and y > 0:
                        print('UP')
                        print('LAUNCH ALGORITHM')
                        softAss_detAnnealing_3().start()

                    if x > 0 and y < 0:
                        print ('RIGHT')
                    if x < 0 and y > 0:
                        for i in range (0, len(bpy.data.curves['Stroke'].splines)):
                            self.remove_spline()
                        print ('LEFT')
                    if x < 0 and y < 0:
                        print ('DOWN')
                        self.remove_spline()

                    self.state = State.IDLE


            elif self.state == State.DECISIONAL:
                print("Decisional")
                bpy.data.objects["Text.R"].data.body = "Selection\n " + self.objToControll + "-" + self.boneToControll


                # Compute the nearest object
                try:
                    self.objToControll, self.boneToControll = self.getClosestItem(True)
                except:
                    print("Error during selection")
                global currObject
                global currBone
                currObject = self.objToControll
                currBone = self.boneToControll
                print("Current obj:", self.objToControll, self.boneToControll)

                if ctrl_state.ulButtonPressed != 4:
                    # print("touch button released")
                    self.changeSelection(self.objToControll, self.boneToControll, True)

                    self.state = State.IDLE

            elif self.state == State.INTERACTION_LOCAL:
                bpy.data.objects["Text.R"].data.body = "Interaction\n" + self.objToControll + "-" + self.boneToControll + "\n" + self.axes[self.curr_axes_r]

                ## Controll object scale
                if self.objToControll == self.objToControll_l and self.boneToControll == self.boneToControll_l and ctrl_state_l.ulButtonPressed == 8589934592:
                    if self.boneToControll != "":
                        self.initial_scale = copy.deepcopy(bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale)
                    else:
                        self.initial_scale = copy.deepcopy(bpy.data.objects[self.objToControll].scale)

                    self.diff_distance = self.computeTargetObjDistance("Controller.L", "", True)


                    self.state = State.SCALING
                    self.state_l = StateLeft.SCALING

                else:
                    if self.boneToControll != "":
                        ## The object to move is a bone
                        bone = bpy.data.objects[self.objToControll]
                        pbone = bone.pose.bones[self.boneToControll]
                        scale = copy.deepcopy(pbone.scale)
                        translationMatrix = Matrix(((0.0, 0.0, 0.0, self.diff_loc[0]),
                                                    (0.0, 0.0, 0.0, self.diff_loc[1]),
                                                    (0.0, 0.0, 0.0, self.diff_loc[2]),
                                                    (0.0, 0.0, 0.0, 1.0)))
                        diff_rot_matr = self.diff_rot.to_matrix()
                        pbone.matrix = (ctrl.matrix_world + translationMatrix) * diff_rot_matr.to_4x4()
                        pbone.scale = scale

                        self.applyConstraint(True)


                    else:
                        ## The object to move is a mesh
                        bpy.data.objects[
                            self.objToControll].rotation_quaternion = ctrl.rotation_quaternion * self.diff_rot
                        bpy.data.objects[self.objToControll].location = ctrl.location + self.diff_loc

                        self.applyConstraint(True)

                if (ctrl_state.ulButtonPressed == 8589934596):
                    print("INTERACTION_LOCAL -> CHANGE_AXIS")
                    self.state = State.CHANGE_AXES

                if (ctrl_state.ulButtonPressed != 8589934592 and ctrl_state.ulButtonPressed != 8589934596):
                    # print("grillet released")
                    self.state = State.IDLE

            elif self.state == State.CHANGE_AXES:
                if (ctrl_state.ulButtonPressed == 8589934592):
                    self.curr_axes_r += 1
                    if self.curr_axes_r >= len(self.axes):
                        self.curr_axes_r = 0
                    self.curr_axes_l = 0
                    self.state = State.INTERACTION_LOCAL

                if (ctrl_state.ulButtonPressed == 0):
                    self.state = State.IDLE

            elif self.state == State.NAVIGATION_ENTER:
                bpy.data.objects["Text.R"].data.body = "Navigation\n "
                bpy.data.objects["Text.L"].data.body = "Navigation\n "
                if ctrl_state.ulButtonPressed != 2:
                    #bpy.data.textures['Texture.R'].image = bpy.data.images['Nav-R.png']
                    #bpy.data.textures['Texture.L'].image = bpy.data.images['Hand-L.png']
                    self.state = State.NAVIGATION
                    self.state_l = StateLeft.NAVIGATION

            elif self.state == State.NAVIGATION_EXIT:
                if ctrl_state.ulButtonPressed != 2:
                    print("NAVIGATION -> IDLE")
                    #bpy.data.textures['Texture.R'].image = bpy.data.images['Perf-R.png']
                    #bpy.data.textures['Texture.L'].image = bpy.data.images['Ctrl-L.png']
                    self.state = State.IDLE
                    self.state_l = StateLeft.IDLE

            elif self.state == State.NAVIGATION:
                if ctrl_state.ulButtonPressed == 4294967296:
                    x, y = ctrl_state.rAxis[0].x, ctrl_state.rAxis[0].y
                    if (x > -0.3 and x < 0.3 and y < -0.8):
                        print("ZOOM_OUT")
                        camObjDist = bpy.data.objects["Origin"].location - camera.location
                        if self.objToControll != "":
                            camObjDist = bpy.data.objects[self.objToControll].location - camera.location
                        camera.location -= camObjDist

                        scale_factor = camera.scale[0]
                        scale_factor = scale_factor * 2
                        camera.scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.R"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.L"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        self.zoom = scale_factor
                        self.state = State.ZOOM_IN

                    if (x > -0.3 and x < 0.3 and y > 0.8):
                        print("ZOOM_IN")
                        camObjDist = bpy.data.objects["Origin"].location - camera.location
                        if self.objToControll != "":
                            camObjDist = bpy.data.objects[self.objToControll].location - camera.location
                        camObjDist = camObjDist / 2
                        camera.location += camObjDist

                        scale_factor = camera.scale[0]
                        scale_factor = scale_factor / 2
                        camera.scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.R"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.L"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        self.zoom = scale_factor
                        self.state = State.ZOOM_OUT

                if (ctrl_state.ulButtonPressed == 8589934592):
                    print('Camera rot: ', camera.rotation_quaternion)
                    self.diff_rot = ctrl.rotation_quaternion.inverted() * camera.rotation_quaternion
                    print('Diff:       ', self.diff_rot)
                    # self.diff_loc = camera.location - ctrl.location
                    self.diff_trans_matrix = bpy.data.objects['Camera'].matrix_world * bpy.data.objects[
                        'Origin'].matrix_world

                    self.rotFlag = False
                    self.state = State.CAMERA_ROT_CONT

                if (ctrl_state.ulButtonPressed == 4):
                    self.diff_loc = copy.deepcopy(ctrl.location)
                    self.state = State.CAMERA_MOVE_CONT

                if (ctrl_state.ulButtonPressed == 2):
                    self.state = State.NAVIGATION_EXIT

            elif self.state == State.ZOOM_IN:
                if (ctrl_state.ulButtonPressed != 4294967296):
                    self.state = State.NAVIGATION

            elif self.state == State.ZOOM_OUT:
                if (ctrl_state.ulButtonPressed != 4294967296):
                    self.state = State.NAVIGATION

            elif self.state == State.CAMERA_MOVE_CONT:
                camera.location = camera.location + (self.diff_loc - ctrl.location)

                if ctrl_state.ulButtonPressed != 4:
                    self.state = State.NAVIGATION

            elif self.state == State.CAMERA_ROT_CONT:

                if (ctrl_state.ulButtonPressed != 8589934592):
                    self.rotFlag = True
                    self.state = State.NAVIGATION

            elif self.state == State.SCALING:
                currDist = self.computeTargetObjDistance("Controller.L", "", True)
                offset = (currDist - self.diff_distance) / 10
                if self.boneToControll != "":
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale = self.initial_scale
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale[0] += offset
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale[1] += offset
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale[2] += offset

                else:
                    bpy.data.objects[self.objToControll].scale = self.initial_scale
                    bpy.data.objects[self.objToControll].scale[0] += offset
                    bpy.data.objects[self.objToControll].scale[1] += offset
                    bpy.data.objects[self.objToControll].scale[2] += offset


                # Exit from Scaling state
                if (ctrl_state.ulButtonPressed != 8589934592):
                    if (ctrl_state_l.ulButtonPressed != 8589934592):
                        self.state = State.IDLE
                        self.state_l = StateLeft.IDLE


            ########## Left_Controller_States ##########

            if self.state_l == StateLeft.IDLE:
                bpy.data.objects["Text.L"].data.body = "Idle\n" + self.objToControll_l + "-" + self.boneToControll_l

                ## TIMELINE NAVIGATION
                if ctrl_state_l.ulButtonPressed == 4294967296:
                    x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y
                    print(x,y)

                # DECISIONAL
                if (ctrl_state_l.ulButtonPressed == 4):
                    print("IDLE -> DECISIONAL")
                    print ("DECISIONAL ENTER: ", self.objToControll_l, self.boneToControll_l)
                    self.changeSelection(self.objToControll_l, self.boneToControll_l, False)
                    self.state_l = StateLeft.DECISIONAL

                # INTERACTION_LOCAL
                if (ctrl_state_l.ulButtonPressed == 8589934592 and self.objToControll_l != ""):
                    print("IDLE -> INTERACTION LOCAL")
                    self.state_l = StateLeft.INTERACTION_LOCAL
                    self.curr_axes_l = 0

                    if self.boneToControll_l != "":
                        self.diff_rot_l = ctrl_l.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].matrix.to_quaternion()
                        self.diff_loc_l = bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].matrix.to_translation() - ctrl_l.location
                        self.initial_loc_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].location)
                        bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].rotation_mode = 'XYZ'
                        self.initial_rot_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].rotation_euler)
                        bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].rotation_mode = 'QUATERNION'

                    else:
                        self.diff_rot_l = ctrl_l.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll_l].rotation_quaternion
                        self.diff_loc_l = bpy.data.objects[self.objToControll_l].location - ctrl_l.location
                        self.initial_loc_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].location)
                        bpy.data.objects[self.objToControll_l].rotation_mode = 'XYZ'
                        self.initial_rot_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].rotation_euler)
                        bpy.data.objects[self.objToControll_l].rotation_mode = 'QUATERNION'

            elif self.state_l == StateLeft.INTERACTION_LOCAL:
                bpy.data.objects["Text.L"].data.body = "Interaction\n" + self.objToControll_l + "-" + self.boneToControll_l + "\n" + self.axes[self.curr_axes_l]


                if self.objToControll == self.objToControll_l \
                        and self.boneToControll == self.boneToControll_l \
                        and ctrl_state.ulButtonPressed == 8589934592\
                        and self.state != State.CAMERA_ROT_CONT \
                        and self.state != State.NAVIGATION:
                    self.state_l = StateLeft.SCALING
                    self.state = State.SCALING

                else:
                    if self.boneToControll_l != "":
                        ## The object to move is a bone
                        bone = bpy.data.objects[self.objToControll_l]
                        pbone = bone.pose.bones[self.boneToControll_l]
                        scale = copy.deepcopy(pbone.scale)
                        translationMatrix = Matrix(((0.0, 0.0, 0.0, self.diff_loc_l[0]),
                                                    (0.0, 0.0, 0.0, self.diff_loc_l[1]),
                                                    (0.0, 0.0, 0.0, self.diff_loc_l[2]),
                                                    (0.0, 0.0, 0.0, 1.0)))
                        diff_rot_matr = self.diff_rot_l.to_matrix()
                        pbone.matrix = (ctrl_l.matrix_world + translationMatrix) * diff_rot_matr.to_4x4()
                        pbone.scale = scale
                        self.applyConstraint(False)

                    else:
                        ## The object to move is a mesh
                        bpy.data.objects[self.objToControll_l].rotation_quaternion = ctrl_l.rotation_quaternion * self.diff_rot_l
                        bpy.data.objects[self.objToControll_l].location = ctrl_l.location + self.diff_loc_l
                        self.applyConstraint(False)


                if (ctrl_state_l.ulButtonPressed==8589934596):
                    print("INTERACTION_LOCAL -> CHANGE_AXIS")
                    self.state_l = StateLeft.CHANGE_AXES

                if (ctrl_state_l.ulButtonPressed != 8589934592 and ctrl_state_l.ulButtonPressed != 8589934596):
                    self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.NAVIGATION:
                if (ctrl_state_l.ulButtonPressed == 4294967296):
                    x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y
                    print (x,y)

            elif self.state_l == StateLeft.CHANGE_AXES:
                if (ctrl_state_l.ulButtonPressed==8589934592):
                    self.curr_axes_l+=1
                    if self.curr_axes_l>=len(self.axes):
                        self.curr_axes_l=0
                    self.curr_axes_r=0
                    print(self.curr_axes_l)
                    print("CHANGE_AXIS -> INTERACTION_LOCAL")
                    self.state_l = StateLeft.INTERACTION_LOCAL

                if (ctrl_state_l.ulButtonPressed==0):
                    # print("grillet released")
                    self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.SCALING:

                # Exit from Scaling state
                if (ctrl_state_l.ulButtonPressed != 8589934592):
                    if (ctrl_state.ulButtonPressed != 8589934592):
                        self.state = State.IDLE
                        self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.DECISIONAL:
                bpy.data.objects["Text.L"].data.body = "Selection\n" + self.objToControll_l + "-" + self.boneToControll_l

                # Compute the nearest object
                self.objToControll_l, self.boneToControll_l = self.getClosestItem(False)
                global currObject_l
                global currBone_l
                currObject_l = self.objToControll_l
                currBone_l = self.boneToControll_l
                print("Current obj:", self.objToControll_l, self.boneToControll_l)

                if ctrl_state_l.ulButtonPressed != 4:
                    self.changeSelection(self.objToControll_l, self.boneToControll_l, True)
                    self.state_l = StateLeft.IDLE
                    print ("DECISIONAL -> IDLE")

            super(OpenVR, self).loop(context)

        except Exception as E:
            self.error("OpenVR.loop", E, False)
            return False

        #if VERBOSE:
        #    print("Left Eye Orientation Raw: " + str(self._eye_orientation_raw[0]))
        #    print("Right Eye Orientation Raw: " + str(self._eye_orientation_raw[1]))

        return True

    def frameReady(self):
        """
        The frame is ready to be sent to the device
        """
        try:
            self._hmd.frameReady()

        except Exception as E:
            self.error("OpenVR.frameReady", E, False)
            return False

        return True

    def reCenter(self):
        """
        Re-center the HMD device

        :return: return True if success
        :rtype: bool
        """
        return self._hmd.reCenter()

    def quit(self):
        """
        Garbage collection
        """
        self._hmd = None
        return super(OpenVR, self).quit()



