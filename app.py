'''
@author: Zhigang Jiang
@time: 2022/05/23
@description:
'''

import gradio as gr
import numpy as np
import os
import torch

from PIL import Image

from utils.logger import get_logger
from config.defaults import get_config
from inference import preprocess, run_one_inference
from models.build import build_model
from argparse import Namespace
import gdown


def down_ckpt(model_cfg, ckpt_dir):
    model_ids = [
        ['src/config/mp3d.yaml', '1o97oAmd-yEP5bQrM0eAWFPLq27FjUDbh'],
        ['src/config/zind.yaml', '1PzBj-dfDfH_vevgSkRe5kczW0GVl_43I'],
        ['src/config/pano.yaml', '1JoeqcPbm_XBPOi6O9GjjWi3_rtyPZS8m'],
        ['src/config/s2d3d.yaml', '1PfJzcxzUsbwwMal7yTkBClIFgn8IdEzI'],
        ['src/config/ablation_study/full.yaml', '1U16TxUkvZlRwJNaJnq9nAUap-BhCVIha']
    ]

    for model_id in model_ids:
        if model_id[0] != model_cfg:
            continue
        path = os.path.join(ckpt_dir, 'best.pkl')
        if not os.path.exists(path):
            logger.info(f"Downloading {model_id}")
            os.makedirs(ckpt_dir, exist_ok=True)
            gdown.download(f"https://drive.google.com/uc?id={model_id[1]}", path, False)


def greet(img_path, pre_processing, weight_name, post_processing, visualization, mesh_format, mesh_resolution):
    args.pre_processing = pre_processing
    args.post_processing = post_processing
    if weight_name == 'mp3d':
        model = mp3d_model
    elif weight_name == 'zind':
        model = zind_model
    else:
        logger.error("unknown pre-trained weight name")
        raise NotImplementedError

    img_name = os.path.basename(img_path).split('.')[0]
    img = np.array(Image.open(img_path).resize((1024, 512), Image.Resampling.BICUBIC))[..., :3]

    vp_cache_path = 'src/demo/default_vp.txt'
    if args.pre_processing:
        vp_cache_path = os.path.join('src/output', f'{img_name}_vp.txt')
        logger.info("pre-processing ...")
        img, vp = preprocess(img, vp_cache_path=vp_cache_path)

    img = (img / 255.0).astype(np.float32)
    run_one_inference(img, model, args, img_name,
                      logger=logger, show=False,
                      show_depth='depth-normal-gradient' in visualization,
                      show_floorplan='2d-floorplan' in visualization,
                      mesh_format=mesh_format, mesh_resolution=int(mesh_resolution))

    return [os.path.join(args.output_dir, f"{img_name}_pred.png"),
            os.path.join(args.output_dir, f"{img_name}_3d{mesh_format}"),
            os.path.join(args.output_dir, f"{img_name}_3d{mesh_format}"),
            vp_cache_path,
            os.path.join(args.output_dir, f"{img_name}_pred.json")]


def get_model(args):
    config = get_config(args)
    down_ckpt(args.cfg, config.CKPT.DIR)
    if ('cuda' in args.device or 'cuda' in config.TRAIN.DEVICE) and not torch.cuda.is_available():
        logger.info(f'The {args.device} is not available, will use cpu ...')
        config.defrost()
        args.device = "cpu"
        config.TRAIN.DEVICE = "cpu"
        config.freeze()
    model, _, _, _ = build_model(config, logger)
    return model


if __name__ == '__main__':
    logger = get_logger()
    args = Namespace(device='cuda', output_dir='src/output', visualize_3d=False, output_3d=True)
    os.makedirs(args.output_dir, exist_ok=True)

    args.cfg = 'src/config/mp3d.yaml'
    mp3d_model = get_model(args)

    args.cfg = 'src/config/zind.yaml'
    zind_model = get_model(args)

    description = "This demo of the project " \
                  "<a href='https://github.com/zhigangjiang/LGT-Net' target='_blank'>LGT-Net</a>. " \
                  "It uses the Geometry-Aware Transformer Network to predict the 3d room layout of an rgb panorama."

    demo = gr.Interface(fn=greet,
                        inputs=[gr.Image(type='filepath', label='input rgb panorama', value='src/demo/pano_demo1.png'),
                                gr.Checkbox(label='pre-processing', value=True),
                                gr.Radio(['mp3d', 'zind'],
                                         label='pre-trained weight',
                                         value='mp3d'),
                                gr.Radio(['manhattan', 'atalanta', 'original'],
                                         label='post-processing method',
                                         value='manhattan'),
                                gr.CheckboxGroup(['depth-normal-gradient', '2d-floorplan'],
                                                 label='2d-visualization',
                                                 value=['depth-normal-gradient', '2d-floorplan']),
                                gr.Radio(['.gltf', '.obj', '.glb'],
                                         label='output format of 3d mesh',
                                         value='.gltf'),
                                gr.Radio(['128', '256', '512', '1024'],
                                         label='output resolution of 3d mesh',
                                         value='256'),
                                ],
                        outputs=[gr.Image(label='predicted result 2d-visualization', type='filepath'),
                                 gr.Model3D(label='3d mesh reconstruction', clear_color=[1.0, 1.0, 1.0, 1.0]),
                                 gr.File(label='3d mesh file'),
                                 gr.File(label='vanishing point information'),
                                 gr.File(label='layout json')],
                        examples=[
                            ['src/demo/pano_demo1.png',  True,  'mp3d', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/mp3d_demo1.png',  False, 'mp3d', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/mp3d_demo2.png',  False, 'mp3d', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/mp3d_demo3.png',  False, 'mp3d', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/zind_demo1.png',  True, 'zind', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/zind_demo2.png',  False, 'zind',  'atalanta', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/zind_demo3.png',  True, 'zind', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/other_demo1.png', False, 'mp3d', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                            ['src/demo/other_demo2.png', True,  'mp3d', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
                        ], title='LGT-Net', allow_flagging="never", cache_examples=False, description=description)

    demo.launch(debug=True, enable_queue=False)
