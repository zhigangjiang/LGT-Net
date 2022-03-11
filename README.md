# LGT-Net
This is PyTorch implementation of our paper "[LGT-Net: Indoor Panoramic Room Layout Estimation with Geometry-Aware Transformer Network](https://arxiv.org/abs/2203.01824)"(CVPR'22). [ [Supplemental Materials](http://file.linkstack.net/other/lgt-net/LGT-Net_supplemental_materials.pdf)]

![network](src/fig/network.png)
# Demo
**We provide a [demo script](https://colab.research.google.com/drive/1mvwXVXGlrkApdMW6_8_prFBHjxwTReA6?usp=sharing) that runs on [Google Clobe](https://colab.research.google.com/) and recommend that you test directly on it.**



# Installation
Install our dependencies:
```shell
pip install -r requirements.txt
```
or try using our development environment
```shell
pip install -r requirements_dev.txt
```
---

If you need 3d to visualize output results, install visualizer (code from [here](https://github.com/fuenwang/360LayoutVisualizer)) dependencies:
```shell
pip install -r visualization/visualizer/requirements.txt
```
macOS need to be aware of this issue:
[Unable to load/find OpenGL](https://github.com/PixarAnimationStudios/USD/issues/1372#issuecomment-716925973).


# Preparing Dataset
### MatterportLayout
Office MatterportLayout dataset is at [here](https://github.com/ericsujw/Matterport3DLayoutAnnotation).

Make sure the dataset files are stored as follows:
```
src/dataset/mp3d
|-- image
|   |-- 17DRP5sb8fy_08115b08da534f1aafff2fa81fc73512.png
|-- label
|   |-- 17DRP5sb8fy_08115b08da534f1aafff2fa81fc73512.json
|-- split
    |-- test.txt
    |-- train.txt
    |-- val.txt

```

---

**Analyse**

|  Split| All |4 Corners |6 Corners |8 Corners |>=10 Corners |
|--|--|--|--|--|--|
| All | 2295  | 1210 | 502 | 309 | 274|
|Train   |1647  | 841 | 371 | 225 | 210 |
|Val   |190   | 108 | 46 | 21 | 15 |
|Test    |458   | 261 | 85 | 63 | 49 |

--
### ZInd
Office ZInd dataset is at [here](https://github.com/zillow/zind).

Make sure the dataset files are stored as follows:
```
src/dataset/zind
|-- 0000
|   |-- panos
|   |   |-- floor_01_partial_room_01_pano_14.jpg
|   |-- zind_data.json
|-- room_shape_simplicity_labels.json
|-- zind_partition.json
```

**Analyse**

|  Split| All |4 Corners |5 Corners |6 Corners |7 Corners |8 Corners|9 Corners |>=10 Corners |Manhattan |No-Manhattan(%) |
|--|--|--|--|--|--|--|--|--|--|--|
|All| 31132 | 17293 |1803 | 7307 | 774 | 2291 | 238 | 1426 |26664 | 4468(14.35%)|
|Train   | 24882 | 13866 |1507 | 5745 | 641 | 1791 | 196 | 1136 |21228 | 3654(14.69%)|
|Val    |  3080 | 1702 | 153  |  745 | 81  |  239 | 22  | 138  |2647 | 433(14.06%)|
|Test    |  3170 | 1725 | 143  |  817 | 52  |  261 | 20  | 152  |2789  | 381(12.02%)|

--

### PanoContext and Stanford 2D-3D
We follow the same preprocessed pano/s2d3d  proposed by [HorizonNet](https://github.com/sunset1995/HorizonNet#dataset).
You also can directly download the dataset file in [here](https://drive.google.com/file/d/164DnSxz6ap8GcytRAPfJlIMvNPaikZEc/view?usp=sharing).

Make sure the dataset files are stored as follows:
```
src/dataset/pano_s2d3d
|-- test
|   |-- img
|   |   |-- camera_0000896878bd47b2a624ad180aac062e_conferenceRoom_3_frame_equirectangular_domain_.png
|   |-- label_cor
|       |-- camera_0000896878bd47b2a624ad180aac062e_conferenceRoom_3_frame_equirectangular_domain_.txt
|-- train
|   |-- img
|   |-- label_cor
|-- valid
    |-- img
    |-- label_cor
     
```
# Downloading Pre-trained Weights
We provide pre-trained weights on individual datasets at [here](https://drive.google.com/drive/folders/1bOZyXeuNnwFEC9nw7EgJUwMiI685obdT?usp=sharing).

- [mp3d/best.pkl](https://drive.google.com/file/d/1o97oAmd-yEP5bQrM0eAWFPLq27FjUDbh/view?usp=sharing): Training on MatterportLayout dataset
- [zind/best.pkl](https://drive.google.com/file/d/1PzBj-dfDfH_vevgSkRe5kczW0GVl_43I/view?usp=sharing): Training on ZInd dataset
- [pano/best.pkl](https://drive.google.com/file/d/1JoeqcPbm_XBPOi6O9GjjWi3_rtyPZS8m/view?usp=sharing): Training on PanoContext(train)+Stanford2D-3D(whole) dataset
- [s2d3d/best.pkl](https://drive.google.com/file/d/1PfJzcxzUsbwwMal7yTkBClIFgn8IdEzI/view?usp=sharing): Training on Stanford2D-3D(train)+PanoContext(whole) dataset
- [ablation_study_full/best.pkl](https://drive.google.com/file/d/1U16TxUkvZlRwJNaJnq9nAUap-BhCVIha/view?usp=sharing): Ablation Study: Ours (full) on MatterportLayout dataset

Make sure the pre-trained weight files are stored as follows:
```
checkpoints
|-- SWG_Transformer_LGT_Net
|   |-- ablation_study_full
|   |   |-- best.pkl
|   |-- mp3d
|   |   |-- best.pkl
|   |-- pano
|   |   |-- best.pkl
|   |-- s2d3d
|   |   |-- best.pkl
|   |-- zind
|       |-- best.pkl
```
# Evaluation
You can evaluate by executing the following command:

- MatterportLayout dataset
    ```shell
    python main.py --cfg src/config/mp3d.yaml --mode test --need_rmse
    ```
- ZInd dataset
    ```shell
    python main.py --cfg src/config/zind.yaml --mode test --need_rmse
    ```
- PanoContext dataset
  ```shell
  python main.py --cfg src/config/pano.yaml --mode test --need_cpe --post_processing manhattan --force_cube
- Stanford 2D-3D dataset
    ```shell
    python main.py --cfg src/config/s2d3d.yaml --mode test --need_cpe --post_processing manhattan --force_cube
    ```
    - `--post_processing` type of post-processing approach, 
      we use [DuLa-Net](https://github.com/SunDaDenny/DuLa-Net) post-processing to process `manhattan` constraint, 
      use [DP algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)  to process `atalanta` constraint, default is disabled.
    - `--need_rmse` need to evaluate root mean squared error and delta error, default is disabled.
    - `--need_cpe` need to evaluate corner error and pixel error, default is disabled.
    - `--force_cube` force cube shape when evaluating, default is disabled.
    - `--wall_num` different corner number to evaluate, default is all.
    - `--save_eval` save the visualization evaluating results of each panorama, 
      the output results locate in the corresponding checkpoint directory 
      (e.g., `checkpoints/SWG_Transformer_LGT_Net/mp3d/results/test`), default is disabled.

# Training
Execute the following commands to train  (e.g., MatterportLayout dataset):
```shell
python main.py --cfg src/config/mp3d.yaml --mode train
```
You can copy and modify the configuration in `YAML` file for other training.

# Inference
We provide an inference script (`inference.py`) that you can
try to predict your panoramas by executing the following command (e.g., using pre-trained weights of MatterportLayout dataset):
```shell
python inference.py --cfg src/config/mp3d.yaml --img_glob src/demo/demo1.png --output_dir src/output --post_processing manhattan
```
It will output json files(`xxx_pred.json`, format is the same as [PanoAnnotator](https://github.com/SunDaDenny/PanoAnnotator)) and visualization images (`xxx_pred.png`) under **output_dir**.
visualization image:
![pred](src/fig/demo1_pred.png)

- `--img_glob` a panorama path or directory path for prediction.

- `--post_processing` If `manhattan` is selected,
we will preprocess the panorama so that the vanishing points are
aligned with the axes for post-processing. Note that after preprocessing 
our predicted results will not align with your input panoramas, 
you can use the output file (`vp.txt`) of vanishing points to reverse align them manually.

- `--visualize_3d` 3D visualization of output results (need install dependencies and GUI desktop environment).
