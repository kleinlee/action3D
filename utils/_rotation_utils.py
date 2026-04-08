import math
from math import cos,sin,radians,atan2,sqrt,degrees
import numpy as np
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return radians(included_angle)

import pandas as pd
def AxisAngle(vec0, vec1):
    axis_R = np.cross(vec0, vec1)
    angle_R = np.linalg.norm(axis_R) / (np.linalg.norm(vec0) * np.linalg.norm(vec1))
    if abs(angle_R) > 1:
        angle_R = angle_R/abs(angle_R)
    angle_R = np.arcsin(angle_R)
    # if pd.isna(angle_R):
    #     print("XAXAXAXAXAXAXAXAXAXA ", vec0, vec1, axis_R, np.linalg.norm(axis_R) / (np.linalg.norm(vec0) * np.linalg.norm(vec1)), angle_R)
    sign_cos = np.dot(vec0, vec1)
    if sign_cos < 0:
        angle_R = np.sign(angle_R) * (np.pi - abs(angle_R))

    # angle_R = angle(vec0, vec1)
    axis_R = axis_R / np.linalg.norm(axis_R)
    # print("AAAA ", axis_R, angle_R)
    return [axis_R, angle_R]


def RotationMatrix(tmp):
    u,angle = tmp[0], tmp[1]
    rotatinMatrix = np.zeros([3, 3])
    rotatinMatrix[0, 0] = math.cos(angle) + u[0] * u[0] * (1 - math.cos(angle))
    rotatinMatrix[0, 1] = -u[2] * math.sin(angle) + u[0] * u[1] * (1 - math.cos(angle))
    rotatinMatrix[0, 2] =  u[1] * math.sin(angle) + u[0] * u[2] * (1 - math.cos(angle))
    rotatinMatrix[1, 0] =  u[2] * math.sin(angle) + u[0] * u[1] * (1 - math.cos(angle))
    rotatinMatrix[1, 1] = math.cos(angle) + u[1] * u[1] * (1 - math.cos(angle))
    rotatinMatrix[1, 2] = -u[0] * math.sin(angle) + u[1] * u[2] * (1 - math.cos(angle))
    rotatinMatrix[2, 0] = -u[1] * math.sin(angle) + u[0] * u[2] * (1 - math.cos(angle))
    rotatinMatrix[2, 1] =  u[0] * math.sin(angle) + u[1] * u[2] * (1 - math.cos(angle))
    rotatinMatrix[2, 2] = math.cos(angle) + u[2] * u[2] * (1 - math.cos(angle))
    # print("########",  np.linalg.det(rotatinMatrix))
    return rotatinMatrix

# def EulurAngle(matRotate, isDegrees = True):
#     tmp = sqrt(matRotate[0, 0]**2 + matRotate[1, 0]**2)
#     singular = (tmp < 1e-6)
#     if not singular:
#         angle_x = atan2(matRotate[2, 1], matRotate[2, 2])
#         angle_y = atan2(-matRotate[2, 0], tmp)
#         angle_z = atan2(matRotate[1, 0], matRotate[0, 0])
#     else:
#         angle_x = atan2(-matRotate[1, 2], matRotate[1, 1])
#         angle_y = atan2(-matRotate[2, 0], tmp)
#         angle_z = 0
#
#     # angle_x = -atan2(matRotate[1, 2], matRotate[2, 2])
#     # angle_y = -math.asin(matRotate[0, 2])
#     # angle_z = -atan2(matRotate[0, 1], matRotate[0, 0])
#
#     # print(angle_x,angle_y,angle_z)
#     if isDegrees:
#         return np.array([degrees(angle_x),degrees(angle_y),degrees(angle_z)])
#     else:
#         return np.array([angle_x,angle_y,angle_z])


def EulurAngle(matRotate, isDegrees=True, restrict = False):
    c2 = sqrt(matRotate[0, 0] ** 2 + matRotate[1, 0] ** 2)
    singular = (c2 < 1e-6)
    if abs(c2) > 1e-6:
        angle_x = atan2(matRotate[2, 1], matRotate[2, 2])
        angle_y = atan2(-matRotate[2, 0], c2)
        angle_z = atan2(matRotate[1, 0], matRotate[0, 0])
        # print("222222222222222 ", matRotate[0, 0], cos(angle_y)*cos(angle_z))
        # print("222222222222222 ", matRotate[1, 0], cos(angle_y)*sin(angle_z))
        # print("222222222222222 ", matRotate[2, 1], sin(angle_x) * cos(angle_y))
        # print("222222222222222 ", matRotate[2, 2], cos(angle_x) * cos(angle_y))
        # print("222222222222222 ", matRotate[2, 0])
        # print("222222222222222 ", matRotate[0, 1], -cos(angle_x)*sin(angle_z) + sin(angle_x)*sin(angle_y)*cos(angle_z))
        # print("222222222222222 ", matRotate[0, 2], sin(angle_x) * sin(angle_z) + cos(angle_x) * sin(angle_y) * cos(angle_z))
        # print("222222222222222 ", matRotate[1, 1], cos(angle_x) * cos(angle_z) + sin(angle_x) * sin(angle_y) * sin(angle_z))
        # print("222222222222222 ", matRotate[1, 2], -sin(angle_x) * cos(angle_z) + cos(angle_x) * sin(angle_y) * sin(angle_z))

        if restrict and abs(angle_z) > np.pi/2:
            print("XXXXXXXXXXX")
            c2 = -1*c2
            angle_x = atan2(matRotate[2, 1]*(-1), matRotate[2, 2]*(-1))
            angle_y = atan2(-matRotate[2, 0], c2)
            angle_z = atan2(matRotate[1, 0]*(-1), matRotate[0, 0]*(-1))

    else:
        angle_x = atan2(-matRotate[1, 2], matRotate[1, 1])
        angle_y = np.pi/2
        angle_z = 0

    # print(angle_x,angle_y,angle_z)
    if isDegrees:
        return np.array([degrees(angle_x), degrees(angle_y), degrees(angle_z)])
    else:
        return np.array([angle_x, angle_y, angle_z])
def RotateAngle2Matrix(tmp):   #tmp为xyz的旋转角,角度值
    tmp = [radians(i) for i in tmp]
    matX = np.array([[1.0,          0,            0],
                     [0.0,          cos(tmp[0]), -sin(tmp[0])],
                     [0.0,          sin(tmp[0]),  cos(tmp[0])]])
    matY = np.array([[cos(tmp[1]),  0,            sin(tmp[1])],
                     [0.0, 1, 0],
                     [-sin(tmp[1]),  0,            cos(tmp[1])]])
    matZ = np.array([[cos(tmp[2]), -sin(tmp[2]),  0],
                     [sin(tmp[2]),  cos(tmp[2]),  0],
                     [0, 0, 1]])
    matRotate = np.matmul(matZ, matY)
    matRotate = np.matmul(matRotate, matX)
    return matRotate