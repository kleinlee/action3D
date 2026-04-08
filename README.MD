# 3D Pose to FBX 项目

## 项目简介
本项目包含视频动作捕捉、动作文件规范化及pyrender渲染演示、FBX动作加工。方便大家自主设计数字人动作。保留了AI生成粗略动作的高效和人工干预的精细度，提供了业界规范的处理流程。半天时间即可完成一套20+动作的数字人制作。

本项目已去除复杂依赖，对Python版本和PyTorch无严格要求。
- 展示及讲解视频在 Bilibili 上
## 未来计划
- 在3D高斯喷溅模型上应用动作
- 更多动作库，包括男女、舞蹈
- 大模型实时动作交互

## 准备工作
1. **下载模型文件**
   - 从 [Hugging Face](https://huggingface.co/waanqii/SMPLest-X/tree/main) 下载 SMPLest-X 模型并放置在 `SMPLest_X/pretrained_models` 目录下
   - 下载 SMPLX 相关文件 `human_model_files` 并放置到 `human_models/human_model_files` 目录：
     - 官方下载：[SMPL-X](https://smpl-x.is.tue.mpg.de/) 和 [SMPL](https://smpl.is.tue.mpg.de/) 身体模型
     - 国内镜像：[ModelScope](https://www.modelscope.cn/models/Damo_XR_Lab/LHMPP-Prior/tree/master/human_model_files)

## 使用步骤

### 1. 生成空闲动作
生成空闲动作作为数字人动作基准，生成的 obj 模型和视频在 `idle_pose` 目录下：

```bash
python 1_generate_idle_pose.py
```

### 2. 视频动作捕捉
使用最新的 SMPLest-X 模型对视频进行动作捕捉。可使用自定义视频，会对输入的视频统一处理并生成 json 动作文件：

```bash
cd SMPLest_X
python demo.py VIDEO_PATHS
```

其中 `VIDEO_PATHS` 是视频文件路径或通配符模式，例如：

### 3. 验证动作效果
对输入的 json 动作文件依次渲染为 mp4 视频文件：

```bash
# 验证动作效果
python render.py --input_paths INPUT_PATH --OUTPUT_VIDEO_PATH OUTPUT_PATH --smooth True
```

参数说明：
- `--input_paths`：输入 JSON 文件目录，默认为 `video_clips`
- `--OUTPUT_VIDEO_PATH`：输出视频目录，默认为 `output_videos`
- `--smooth`：是否平滑动画数据，默认为 `True`

### 4. 生成 Maya/3D Max 动作文件

```bash
python 4_generate_maya_action.py --input_paths INPUT_PATH --output_paths OUTPUT_PATH
```

参数说明：
- `--input_paths`：输入 JSON 文件目录，默认为 `video_clips`
- `--output_paths`：输出 FBX 文件目录，默认为 `video_clips`

### 5. 修改 FBX 模型并渲染
修改 FBX 模型并以 ASCII 格式保存后再次渲染：

```bash
python 5_fbx_processor.py INPUT_PATH
```

参数说明：
- `INPUT_PATH`：FBX 文件路径
