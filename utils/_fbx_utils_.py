import numpy as np
import os
from utils.smplx_utils import smplx_parents_index_list, smplx_joint_names
frame_time = 1539538600

def writeCurveInfo(fp, id, values):
    fp.write("	AnimationCurve: {}, \"AnimCurve::\", \"\"\n".format(id))
    fp.write("	{\n")
    fp.write("		Default: 0\n")
    fp.write("		KeyVer: 4009\n")
    fp.write("		KeyTime: *{} \n".format(len(values)))
    fp.write("		{\n")
    frame_time_list = [str(tmp*frame_time) for tmp in range(len(values))]
    fp.write("			a:{}\n".format(",".join(frame_time_list)))
    fp.write("		}\n")
    fp.write("	    KeyValueFloat: *{} \n".format(len(values)))
    fp.write("		{\n")
    fp.write("			a:{}\n".format(",".join(list(values.astype(str)))))
    fp.write("		}\n")
    # #这个是分为三段
    # fp.write("		;KeyAttrFlags: Cubic|TangeantUser|WeightedRight|WeightedNextLeft, Constant|ConstantStandard|WeightedRight, Constant|ConstantStandard\n")
    # fp.write("		KeyAttrFlags: *3 {\n")
    # fp.write("			a:8456,8456,1032\n")
    # fp.write("		}\n")
    # fp.write("		;KeyAttrDataFloat: RightAuto:0, NextLeftAuto:0; RightAuto:0, NextLeftAuto:0.538387; RightSlope:0.538387, "
    #          "NextLeftSlope:0, RightWeight:0.333333, NextLeftWeight:0.333333, RightVelocity:0, NextLeftVelocity:0\n")
    # fp.write("		KeyAttrDataFloat: *12 {\n")
    # fp.write("			a: 0, 0, 218434821, 0, 0, -1073644511, 218434821, 0, -1073644512, 0, 218434821, 0\n")
    # fp.write("		}\n")
    # fp.write("		KeyAttrRefCount: *3 {\n")
    # fp.write("		    a: {}, 1, 1\n".format(len(values)-2))
    # fp.write("		}\n")
    fp.write("		;KeyAttrFlags: Cubic|TangeantAuto|GenericTimeIndependent\n")
    fp.write("		KeyAttrFlags: *1 {\n")
    fp.write("			a:8456\n")
    fp.write("		}\n")
    fp.write("		;KeyAttrDataFloat: RightAuto:0, NextLeftAuto: 0\n")
    fp.write("		KeyAttrDataFloat: *4 {\n")
    fp.write("			a: 0, 0, 218434821, 0\n")
    fp.write("		}\n")
    fp.write("		KeyAttrRefCount: *1 {\n")
    fp.write("		    a: {}\n".format(len(values)))
    fp.write("		}\n")
    fp.write("	}\n")
    fp.write("	\n")



def writeObjects(fp, rotate_list, bone_name_list, frame_num,  geometry = None):
    #Model  NodeAttribute  AnimationCurve  AnimationCurveNode
    fp.write("Objects: {\n")
    if geometry is not None:
        verts3D, face_mesh, weights_data, mesh_world_matrix, j_t_pose, j_t_pose_local = geometry

        fp.write("	Model: 20210816, \"Model::CC_Base_Body\", \"Mesh\" \n")
        fp.write("	{\n")
        fp.write("		Version: 232\n")
        fp.write("		Properties70:  {\n")
        fp.write("			P: \"RotationActive\", \"bool\", \"\", \"\",1\n")
        fp.write("			P: \"InheritType\", \"enum\", \"\", \"\",1\n")
        fp.write("			P: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0\n")
        fp.write("			P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n")
        fp.write("			P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",{},{},{}\n".format(0,0,0))
        fp.write("			P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",{},{},{}\n".format(0,0,0))
        fp.write("			P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1\n")
        fp.write("		}\n")
        fp.write("		Shading: Y\n")
        fp.write("		Culling: \"CullingOff\"\n")
        fp.write("	}\n")

        fp.write("	Geometry: 202108160, \"Geometry::\", \"Mesh\" {\n")
        fp.write("		Vertices: *{} {{\n".format(len(verts3D)*3))
        fp.write("			a: ")
        fp.write(",".join([",".join(map(str,vert)) for vert in verts3D]))
        fp.write("\n")

        fp.write("		} \n")
        fp.write("		PolygonVertexIndex: *{} {{\n".format(len(face_mesh)*3))
        fp.write("			a: ")

        fbx_indices = []
        for face in face_mesh:
            # face 是一个包含 3 个索引的元组或列表，例如 [0, 1, 2]
            for i, idx in enumerate(face):
                if i == 2:  # 如果是三角形的第 3 个（最后一个）顶点
                    # FBX 负数转换规则：正常索引 * -1 - 1
                    # 同时确保输入的 idx 是整数
                    fbx_indices.append(str(int(idx) * -1 - 1))
                else:
                    fbx_indices.append(str(int(idx)))
        
        fp.write(",".join(fbx_indices))
        fp.write("\n")
        fp.write("		} \n")
        fp.write("	} \n")

        # with open(r"geometry", "r") as file0:
        #     header = file0.readlines()
        #     for i in header:
        #         fp.write(i)
        
        # 开启蒙皮数据
        skin_id = 202108161
        fp.write("	Deformer: {}, \"Deformer::Skin\", \"Skin\" {{\n".format(skin_id))
        fp.write("		Version: 101\n")
        fp.write("		Link_DeformAcuracy: 50\n")
        fp.write("		SkinningType: \"Linear\"\n")
        fp.write("	}\n")

        # 4. 写入 Cluster（包含正确矩阵计算）
        if weights_data is not None and j_t_pose is not None:
            # mesh_world 强制为单位矩阵，无需改变
            mesh_world = np.eye(4)
            
            for bone_idx, bone_name in enumerate(bone_name_list):
                indices, weights = weights_data[bone_idx]
                if len(indices) == 0:
                    continue
                
                cluster_id = 2021081620 + bone_idx
                
                # 构建骨骼世界矩阵 (4x4)
                bone_tpose = j_t_pose[bone_idx]
                bone_world_matrix = np.eye(4)
                bone_world_matrix[0,0] = 1
                bone_world_matrix[1,1] = 1
                bone_world_matrix[2,2] = 1
                bone_world_matrix[:3, 3] = -bone_tpose[:3]
                
                # 计算 Transform 矩阵
                bone_world_inv = np.linalg.inv(bone_world_matrix) # 计算骨骼世界矩阵的逆
                transform_link_matrix = bone_world_inv @ mesh_world
                transform_matrix = bone_world_matrix
                
                # 写入 Cluster
                fp.write("	Deformer: {}, \"SubDeformer::Cluster {}\", \"Cluster\" {{\n".format(
                    cluster_id, bone_name))
                fp.write("		Version: 100\n")
                fp.write("		Indexes: *{} {{\n".format(len(indices)))
                fp.write("			a: {}\n".format(",".join(map(str, indices))))
                fp.write("		}\n")
                fp.write("		Weights: *{} {{\n".format(len(weights)))
                # 权重保留6位小数避免精度问题
                fp.write("			a: {}\n".format(",".join(map(str, [round(float(w), 6) for w in weights]))))
                fp.write("		}\n")
                
                # Transform: Geometry Space -> Bone Space
                transform_flat = transform_matrix.T.flatten()
                fp.write("		Transform: *16 {\n")
                fp.write("			a: {}\n".format(",".join(map(str, [round(float(x), 10) for x in transform_flat]))))
                fp.write("		}\n")
                
                # TransformLink: Bone World Matrix in Bind Pose
                transform_link_flat = transform_link_matrix.T.flatten()
                fp.write("		TransformLink: *16 {\n")
                fp.write("			a: {}\n".format(",".join(map(str, [round(float(x), 10) for x in transform_link_flat]))))
                fp.write("		}\n")
                fp.write("	}\n")


    for index,i in enumerate(bone_name_list):
        fp.write("	NodeAttribute: {}0, \"NodeAttribute::\", \"LimbNode\" \n".format(100+index))
        fp.write("	{\n")
        fp.write("		TypeFlags: \"Skeleton\"\n")
        fp.write("	}\n")

    for index,i in enumerate(bone_name_list):
        fp.write("	Model: {}, \"Model::{}\", \"LimbNode\" \n".format(100+index, "CC_Base_" + i))
        fp.write("	{\n")
        fp.write("		Version: 232\n")
        fp.write("		Properties70:  {\n")
        # if index == 0:
        #     fp.write("			P: \"PreRotation\", \"Vector3D\", \"Vector\", \"\",0,-0,0\n")
        #     # fp.write("			P: \"PreRotation\", \"Vector3D\", \"Vector\", \"\",-90,-0,0\n")
        fp.write("			P: \"RotationActive\", \"bool\", \"\", \"\",1\n")
        fp.write("			P: \"InheritType\", \"enum\", \"\", \"\",1\n")
        fp.write("			P: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0\n")
        fp.write("			P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n")
        fp.write("			P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",{},{},{}\n".format(
            j_t_pose_local[index][0], j_t_pose_local[index][1], j_t_pose_local[index][2]))
        # fp.write("			P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",{},{},{}\n".format(
        #     rotate_list[index][0][0], rotate_list[index][0][1], rotate_list[index][0][2]))
        fp.write("			P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1\n")
        fp.write("		}\n")
        fp.write("		Shading: Y\n")
        fp.write("		Culling: \"CullingOff\"\n")
        fp.write("	}\n")

    fp.write("	AnimationStack: 2, \"AnimStack::Take 001\", \"\"\n")
    fp.write("	{\n")
    fp.write("		Properties70: {\n")
    fp.write("			P: \"LocalStop\", \"KTime\", \"Time\", \"\", {}\n".format(frame_num*frame_time))
    fp.write("			P: \"ReferenceStop\", \"KTime\", \"Time\", \"\", {}\n".format(frame_num*frame_time))
    fp.write("		}\n")
    fp.write("	}\n")

    fp.write("	AnimationLayer: 1, \"AnimLayer::BaseLayer\", \"\"{\n")
    fp.write("	}\n")
    for index, i in enumerate(bone_name_list):
        fp.write("	AnimationCurveNode: {}, \"AnimCurveNode::T\", \"\"\n".format(str(100 + index) + "0" + "0"))
        fp.write("	{\n")
        fp.write("		Properties70: {\n")
        fp.write("			P: \"d|X\", \"Number\", \"\", \"A\", {}\n".format(0))
        fp.write("			P: \"d|Y\", \"Number\", \"\", \"A\", {}\n".format(0))
        fp.write("			P: \"d|Z\", \"Number\", \"\", \"A\", {}\n".format(0))
        fp.write("		}\n")
        fp.write("	}\n")
        fp.write("	\n")
    for index, i in enumerate(bone_name_list):
        fp.write("	AnimationCurveNode: {}, \"AnimCurveNode::R\", \"\"\n".format(str(100 + index) + "0" + "1"))
        fp.write("	{\n")
        fp.write("		Properties70: {\n")
        # fp.write("			P: \"d|X\", \"Number\", \"\", \"A\", {}\n".format(rotate_list[index][0][0]))
        # fp.write("			P: \"d|Y\", \"Number\", \"\", \"A\", {}\n".format(rotate_list[index][0][1]))
        # fp.write("			P: \"d|Z\", \"Number\", \"\", \"A\", {}\n".format(rotate_list[index][0][2]))
        fp.write("			P: \"d|X\", \"Number\", \"\", \"A\", {}\n".format(0))
        fp.write("			P: \"d|Y\", \"Number\", \"\", \"A\", {}\n".format(0))
        fp.write("			P: \"d|Z\", \"Number\", \"\", \"A\", {}\n".format(0))
        fp.write("		}\n")
        fp.write("	}\n")
        fp.write("	\n")
    for index, i in enumerate(bone_name_list):
        fp.write("	AnimationCurveNode: {}, \"AnimCurveNode::S\", \"\"\n".format(str(100 + index) + "0" + "2"))
        fp.write("	{\n")
        fp.write("		Properties70: {\n")
        fp.write("			P: \"d|X\", \"Number\", \"\", \"A\", {}\n".format(1))
        fp.write("			P: \"d|Y\", \"Number\", \"\", \"A\", {}\n".format(1))
        fp.write("			P: \"d|Z\", \"Number\", \"\", \"A\", {}\n".format(1))
        fp.write("		}\n")
        fp.write("	}\n")
        fp.write("	\n")
    # curve
    for index, i in enumerate(bone_name_list):
        if i == "BoneRoot":
            continue
        # writeCurveInfo(fp, str(100 + index) + "0" + "0" + "0", np.array([trans_list[index][t][0] for t in range(frame_num)]))
        # writeCurveInfo(fp, str(100 + index) + "0" + "0" + "1", np.array([trans_list[index][t][1] for t in range(frame_num)]))
        # writeCurveInfo(fp, str(100 + index) + "0" + "0" + "2", np.array([trans_list[index][t][2] for t in range(frame_num)]))
        writeCurveInfo(fp, str(100 + index) + "0" + "0" + "0", np.array([0 for t in range(frame_num)]))
        writeCurveInfo(fp, str(100 + index) + "0" + "0" + "1", np.array([0 for t in range(frame_num)]))
        writeCurveInfo(fp, str(100 + index) + "0" + "0" + "2", np.array([0 for t in range(frame_num)]))
        writeCurveInfo(fp, str(100 + index) + "0" + "1" + "0", np.array([float(rotate_list[t][index][0]) for t in range(frame_num)]))
        writeCurveInfo(fp, str(100 + index) + "0" + "1" + "1", np.array([float(rotate_list[t][index][1]) for t in range(frame_num)]))
        writeCurveInfo(fp, str(100 + index) + "0" + "1" + "2", np.array([float(rotate_list[t][index][2]) for t in range(frame_num)]))

    fp.write("}\n")


def writeConnections(fp, bone_name_list, geometry = None):
    fp.write("\n")
    fp.write("; Object connections\n")
    fp.write(";------------------------------------------------------------------\n")
    fp.write("\n")
    fp.write("Connections: {\n")

    fp.write("	;AnimLayer::BaseLayer, AnimStack::Take 001\n")
    fp.write("	C: \"OO\", 1, 2\n")
    fp.write("	\n")

    if geometry is not None:
        verts3D, face_mesh, weights_data, mesh_world_matrix, j_t_pose, j_t_pose_local = geometry
        fp.write("	;Model::CC_Base_Body, Model::RootNode\n")
        fp.write("	C: \"OO\", {}, {}\n".format(20210816, 0))
        fp.write("	\n")
        fp.write("	;Geometry, Model::CC_Base_Body\n")
        fp.write("	C: \"OO\", {}, {}\n".format(202108160, 20210816))
        fp.write("	\n")

        # Skin -> Geometry
        skin_id = 202108161
        fp.write("	;Deformer::Skin, Geometry::\n")
        fp.write("	C: \"OO\", {}, {}\n".format(skin_id, 202108160))
        fp.write("	\n")

        # Clusters -> Skin 和 Model -> Cluster
        if weights_data is not None:
            for bone_idx, bone_name in enumerate(bone_name_list):
                if bone_idx >= len(weights_data):
                    continue
                    
                indices, weights = weights_data[bone_idx]
                if len(indices) == 0:
                    continue
                
                cluster_id = 2021081620 + bone_idx
                model_id = 100 + bone_idx
                
                # Cluster -> Skin (SubDeformer -> Deformer)
                fp.write("	;SubDeformer::Cluster {}, Deformer::Skin\n".format(bone_name))
                fp.write("	C: \"OO\", {}, {}\n".format(cluster_id, skin_id))
                fp.write("	\n")
                
                # Model -> Cluster (Bone -> SubDeformer)
                fp.write("	;Model::{}, SubDeformer::Cluster {}\n".format(
                    "CC_Base_" + bone_name, bone_name))
                fp.write("	C: \"OO\", {}, {}\n".format(model_id, cluster_id))
                fp.write("	\n")


    for index, i in enumerate(bone_name_list):
        i = "CC_Base_" + i
        if index == 0:
            parent_index = -100
            parent_bone_name = "RootNode"
        else:
            parent_index = smplx_parents_index_list[index]
            parent_bone_name = smplx_joint_names[parent_index]
       
        fp.write("	;Model::{}, Model::{}\n".format(i, parent_bone_name))

        
        fp.write("	C: \"OO\", {}, {}\n".format(100+index, 100 + parent_index))
        fp.write("	\n")

        fp.write("	;NodeAttribute::, Model::{}\n".format(i))
        fp.write("	C: \"OO\", {}0, {}\n".format(100 + index, 100 + index))
        fp.write("	\n")


    for index, i in enumerate(bone_name_list):
        i = "CC_Base_" + i
        fp.write("	;AnimCurveNode::R, Model::{}\n".format(i))
        fp.write("	C: \"OP\", {}, {}, \"Lcl Rotation\"\n".format(str(100 + index) + "0" + "1", 100 + index))
        fp.write("	;AnimCurveNode::R, AnimLayer::BaseLayer\n")
        fp.write("	C: \"OO\", {}, 1\n".format(str(100 + index) + "0" + "1"))

    for index, i in enumerate(bone_name_list):
        if i == "BoneRoot":
            continue

        fp.write("	;AnimCurve::, AnimCurveNode::R\n")
        fp.write(
            "	C: \"OP\", {}, {}, \"d|X\"\n".format(str(100 + index) + "0" + "1" + "0", str(100 + index) + "0" + "1"))
        fp.write("	;AnimCurve::, AnimCurveNode::R\n")
        fp.write(
            "	C: \"OP\", {}, {}, \"d|Y\"\n".format(str(100 + index) + "0" + "1" + "1", str(100 + index) + "0" + "1"))
        fp.write("	;AnimCurve::, AnimCurveNode::R\n")
        fp.write(
            "	C: \"OP\", {}, {}, \"d|Z\"\n".format(str(100 + index) + "0" + "1" + "2", str(100 + index) + "0" + "1"))
        fp.write("	\n")
    fp.write("}\n")

def writeHeader(fp, frame_num, frame_time):
    with open(os.path.join(os.path.dirname(__file__), "header"), "r") as file0:
        header = file0.readlines()
        rows = 0
        while rows < len(header):
            i = header[rows]
            if i.find("TimeSpanStop") > -1:
                fp.write("		P: \"TimeSpanStop\", \"KTime\", \"Time\", \"\",{}\n".format(frame_num*frame_time))
                rows = rows + 1
            else:
                fp.write(i)
                rows = rows + 1
def writeTailer(fp, frame_num, frame_time):
    with open(os.path.join(os.path.dirname(__file__), "tailer"), "r") as file0:
        header = file0.readlines()
        rows = 0
        while rows < len(header):
            i = header[rows]
            if i.find("LocalTime") > -1:
                fp.write("		LocalTime: 0,{}\n".format(frame_num*frame_time))
                rows = rows + 1
            elif i.find("ReferenceTime") > -1:
                fp.write("		ReferenceTime: 0,{}\n".format(frame_num*frame_time))
                rows = rows + 1
            else:
                fp.write(i)
                rows = rows + 1
def generateFBX(file_path, animation_data, geometry = None):
    '''
    geometry: (verts3D, faces_mesh, weights_data, mesh_world_matrix, j_t_pose, j_t_pose_local)
    其中weights_data[bone_idx] = (np.array(idxs, dtype=np.int32), np.array(ws, dtype=np.float64))
    其中mesh_world_matrix为网格的世界矩阵，j_t_pose为关节的世界坐标
    其中j_t_pose是55个骨骼的世界坐标
    网格初始是标准的T-pose，骨骼的世界空间矩阵中的旋转应该为单位矩阵
    animation_data是一个帧序列表，每一帧包含（55， 3）的欧拉角信息
    '''
    frame_num = len(animation_data)

    with open(file_path, "w") as f:
        writeHeader(f, frame_num, frame_time)
        writeObjects(f, animation_data, smplx_joint_names, frame_num, geometry)
        writeConnections(f, smplx_joint_names, geometry)
        writeTailer(f, frame_num, frame_time)