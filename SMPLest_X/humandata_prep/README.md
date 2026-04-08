Guide to HumanData and View tools
========================

## What is HumanData?

HumanData is designed to provide a unified format for SMPL/SMPLX datasets to support joint training and evaluation.

The project is maintained in MMHuman3D.
See [detailed info](https://github.com/open-mmlab/mmhuman3d/blob/convertors/docs/human_data.md) for data structure and sample usage.

If you want to create your own humandata file, please refer to the sample below and maintain the similiar structure. Basically it is a big dictionary with some lists or dicts of lists, any dict with the correct structure works (Not necesscarily in `HumanData` class).

## Sample Visualization Script

We provide a simple script to check the annotation and visualize the results. The script will read the annotation from HumanData and render it on the corresponding image using pyrender.

### Download

Download sample here: [Hugging Face](https://huggingface.co/waanqii/SMPLest-X/resolve/main/hd_sample_humandata.zip?download=true)

### Extract
Follow the file structure as in main page. Extract to `data` folder,  the structure should look like this:
```
├── data
│   ├── annot
│   │    └── hd_10sample.npz # sample annotation
│   └── img # original data files
│        └── egocentric_color
```

### Environment
Basically you can directly install pyrender and trimesh to your environment, I tested many platforms without finding confilcts.
CPU version of pytorch is also supported.
```
conda create -n hd_vis python=3.9
conda activate hd_vis
conda install torch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pyrender trimesh numpy opencv-python tqdm smplx
```

### Visualization
Fixed command to for demo sample.
```
python humandata_prep/check.py \ 
    --hd_path data/annot/hd_10sample.npz \ 
    --image_folder data/img \ 
    --output_folder data/vis_output \ 
    --body_model_path human_models/human_model_files 
```
- Rendered image will be saved in the output folder.


## Important Points: when visualizing other humandata files
This section is for those who want to debug or create their own humandata files.

- Check `flat_hand_mean` if is correctly set, for humandata, it shoule be specified in `hd['misc']['flat_hand_mean']` or by default `False`
- Check `gender`
- For some specific datasets, they might provide mesh vertices instead of SMPL/SMPLX parameters, we suggest to fit the mesh to parameters for every instance to maintain the consistency of the visualization. Some of those datasets are:
    - Arctic: They provide `vtemplate` instead of `betas`
    - EHF: They provide mesh files
- Standalone [SMPLX parameters fitting script](https://github.com/open-mmlab/mmhuman3d/blob/convertors/tools/preprocess/fit_shape2smplx.py)
 


