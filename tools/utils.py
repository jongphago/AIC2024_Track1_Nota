import random
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import os
import pickle
import json
from tools.draw_table import create_track_records

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

sources = {
    # AIHub
    "scene_042": sorted(
        [
            os.path.join("frames/val/scene_042", p)
            for p in os.listdir("frames/val/scene_042")
        ]
    ),
    # Val
    # 'scene_041': sorted([os.path.join('/workspace/frames/val/scene_041', p) for p in os.listdir('/workspace/frames/val/scene_041')]),
    # 'scene_042': sorted([os.path.join('/workspace/frames/val/scene_042', p) for p in os.listdir('/workspace/frames/val/scene_042')]),
    # 'scene_043': sorted([os.path.join('/workspace/frames/val/scene_043', p) for p in os.listdir('/workspace/frames/val/scene_043')]),
    # 'scene_044': sorted([os.path.join('/workspace/frames/val/scene_044', p) for p in os.listdir('/workspace/frames/val/scene_044')]),
    # 'scene_045': sorted([os.path.join('/workspace/frames/val/scene_045', p) for p in os.listdir('/workspace/frames/val/scene_045')]),
    # 'scene_046': sorted([os.path.join('/workspace/frames/val/scene_046', p) for p in os.listdir('/workspace/frames/val/scene_046')]),
    # 'scene_047': sorted([os.path.join('/workspace/frames/val/scene_047', p) for p in os.listdir('/workspace/frames/val/scene_047')]),
    # 'scene_048': sorted([os.path.join('/workspace/frames/val/scene_048', p) for p in os.listdir('/workspace/frames/val/scene_048')]),
    # 'scene_049': sorted([os.path.join('/workspace/frames/val/scene_049', p) for p in os.listdir('/workspace/frames/val/scene_049')]),
    # 'scene_050': sorted([os.path.join('/workspace/frames/val/scene_050', p) for p in os.listdir('/workspace/frames/val/scene_050')]),
    # 'scene_051': sorted([os.path.join('/workspace/frames/val/scene_051', p) for p in os.listdir('/workspace/frames/val/scene_051')]),
    # 'scene_052': sorted([os.path.join('/workspace/frames/val/scene_052', p) for p in os.listdir('/workspace/frames/val/scene_052')]),
    # 'scene_053': sorted([os.path.join('/workspace/frames/val/scene_053', p) for p in os.listdir('/workspace/frames/val/scene_053')]),
    # 'scene_054': sorted([os.path.join('/workspace/frames/val/scene_054', p) for p in os.listdir('/workspace/frames/val/scene_054')]),
    # 'scene_055': sorted([os.path.join('/workspace/frames/val/scene_055', p) for p in os.listdir('/workspace/frames/val/scene_055')]),
    # 'scene_056': sorted([os.path.join('/workspace/frames/val/scene_056', p) for p in os.listdir('/workspace/frames/val/scene_056')]),
    # 'scene_057': sorted([os.path.join('/workspace/frames/val/scene_057', p) for p in os.listdir('/workspace/frames/val/scene_057')]),
    # 'scene_058': sorted([os.path.join('/workspace/frames/val/scene_058', p) for p in os.listdir('/workspace/frames/val/scene_058')]),
    # 'scene_059': sorted([os.path.join('/workspace/frames/val/scene_059', p) for p in os.listdir('/workspace/frames/val/scene_059')]),
    # 'scene_060': sorted([os.path.join('/workspace/frames/val/scene_060', p) for p in os.listdir('/workspace/frames/val/scene_060')]),
    # Test
    # 'scene_061': sorted([os.path.join('/workspace/frames/test/scene_061', p) for p in os.listdir('/workspace/frames/test/scene_061')]),
    # 'scene_062': sorted([os.path.join('/workspace/frames/test/scene_062', p) for p in os.listdir('/workspace/frames/test/scene_062')]),
    # 'scene_063': sorted([os.path.join('/workspace/frames/test/scene_063', p) for p in os.listdir('/workspace/frames/test/scene_063')]),
    # 'scene_064': sorted([os.path.join('/workspace/frames/test/scene_064', p) for p in os.listdir('/workspace/frames/test/scene_064')]),
    # 'scene_065': sorted([os.path.join('/workspace/frames/test/scene_065', p) for p in os.listdir('/workspace/frames/test/scene_065')]),
    # 'scene_066': sorted([os.path.join('/workspace/frames/test/scene_066', p) for p in os.listdir('/workspace/frames/test/scene_066')]),
    # 'scene_067': sorted([os.path.join('/workspace/frames/test/scene_067', p) for p in os.listdir('/workspace/frames/test/scene_067')]),
    # 'scene_068': sorted([os.path.join('/workspace/frames/test/scene_068', p) for p in os.listdir('/workspace/frames/test/scene_068')]),
    # 'scene_069': sorted([os.path.join('/workspace/frames/test/scene_069', p) for p in os.listdir('/workspace/frames/test/scene_069')]),
    # 'scene_070': sorted([os.path.join('/workspace/frames/test/scene_070', p) for p in os.listdir('/workspace/frames/test/scene_070')]),
    # 'scene_071': sorted([os.path.join('/workspace/frames/test/scene_071', p) for p in os.listdir('/workspace/frames/test/scene_071')]),
    # 'scene_072': sorted([os.path.join('/workspace/frames/test/scene_072', p) for p in os.listdir('/workspace/frames/test/scene_072')]),
    # 'scene_073': sorted([os.path.join('/workspace/frames/test/scene_073', p) for p in os.listdir('/workspace/frames/test/scene_073')]),
    # 'scene_074': sorted([os.path.join('/workspace/frames/test/scene_074', p) for p in os.listdir('/workspace/frames/test/scene_074')]),
    # 'scene_075': sorted([os.path.join('/workspace/frames/test/scene_075', p) for p in os.listdir('/workspace/frames/test/scene_075')]),
    # 'scene_076': sorted([os.path.join('/workspace/frames/test/scene_076', p) for p in os.listdir('/workspace/frames/test/scene_076')]),
    # 'scene_077': sorted([os.path.join('/workspace/frames/test/scene_077', p) for p in os.listdir('/workspace/frames/test/scene_077')]),
    # 'scene_078': sorted([os.path.join('/workspace/frames/test/scene_078', p) for p in os.listdir('/workspace/frames/test/scene_078')]),
    # 'scene_079': sorted([os.path.join('/workspace/frames/test/scene_079', p) for p in os.listdir('/workspace/frames/test/scene_079')]),
    # 'scene_080': sorted([os.path.join('/workspace/frames/test/scene_080', p) for p in os.listdir('/workspace/frames/test/scene_080')]),
    # 'scene_081': sorted([os.path.join('/workspace/frames/test/scene_081', p) for p in os.listdir('/workspace/frames/test/scene_081')]),
    # 'scene_082': sorted([os.path.join('/workspace/frames/test/scene_082', p) for p in os.listdir('/workspace/frames/test/scene_082')]),
    # 'scene_083': sorted([os.path.join('/workspace/frames/test/scene_083', p) for p in os.listdir('/workspace/frames/test/scene_083')]),
    # 'scene_084': sorted([os.path.join('/workspace/frames/test/scene_084', p) for p in os.listdir('/workspace/frames/test/scene_084')]),
    # 'scene_085': sorted([os.path.join('/workspace/frames/test/scene_085', p) for p in os.listdir('/workspace/frames/test/scene_085')]),
    # 'scene_086': sorted([os.path.join('/workspace/frames/test/scene_086', p) for p in os.listdir('/workspace/frames/test/scene_086')]),
    # 'scene_087': sorted([os.path.join('/workspace/frames/test/scene_087', p) for p in os.listdir('/workspace/frames/test/scene_087')]),
    # 'scene_088': sorted([os.path.join('/workspace/frames/test/scene_088', p) for p in os.listdir('/workspace/frames/test/scene_088')]),
    # 'scene_089': sorted([os.path.join('/workspace/frames/test/scene_089', p) for p in os.listdir('/workspace/frames/test/scene_089')]),
    # 'scene_090': sorted([os.path.join('/workspace/frames/test/scene_090', p) for p in os.listdir('/workspace/frames/test/scene_090')]),
}

result_paths = {
    # AIHub
    "scene_042": "./results/scene_042.txt",
    # Val
    # 'scene_041': './results/scene_041.txt',
    # 'scene_042': './results/scene_042.txt',
    # 'scene_043': './results/scene_043.txt',
    # 'scene_044': './results/scene_044.txt',
    # 'scene_045': './results/scene_045.txt',
    # 'scene_046': './results/scene_046.txt',
    # 'scene_047': './results/scene_047.txt',
    # 'scene_048': './results/scene_048.txt',
    # 'scene_049': './results/scene_049.txt',
    # 'scene_050': './results/scene_050.txt',
    # 'scene_051': './results/scene_051.txt',
    # 'scene_052': './results/scene_052.txt',
    # 'scene_053': './results/scene_053.txt',
    # 'scene_054': './results/scene_054.txt',
    # 'scene_055': './results/scene_055.txt',
    # 'scene_056': './results/scene_056.txt',
    # 'scene_057': './results/scene_057.txt',
    # 'scene_058': './results/scene_058.txt',
    # 'scene_059': './results/scene_059.txt',
    # 'scene_060': './results/scene_060.txt',
    # Test
    # 'scene_061': './results/scene_061.txt',
    # 'scene_062': './results/scene_062.txt',
    # 'scene_063': './results/scene_063.txt',
    # 'scene_064': './results/scene_064.txt',
    # 'scene_065': './results/scene_065.txt',
    # 'scene_066': './results/scene_066.txt',
    # 'scene_067': './results/scene_067.txt',
    # 'scene_068': './results/scene_068.txt',
    # 'scene_069': './results/scene_069.txt',
    # 'scene_070': './results/scene_070.txt',
    # 'scene_071': './results/scene_071.txt',
    # 'scene_072': './results/scene_072.txt',
    # 'scene_073': './results/scene_073.txt',
    # 'scene_074': './results/scene_074.txt',
    # 'scene_075': './results/scene_075.txt',
    # 'scene_076': './results/scene_076.txt',
    # 'scene_077': './results/scene_077.txt',
    # 'scene_078': './results/scene_078.txt',
    # 'scene_079': './results/scene_079.txt',
    # 'scene_080': './results/scene_080.txt',
    # 'scene_081': './results/scene_081.txt',
    # 'scene_082': './results/scene_082.txt',
    # 'scene_083': './results/scene_083.txt',
    # 'scene_084': './results/scene_084.txt',
    # 'scene_085': './results/scene_085.txt',
    # 'scene_086': './results/scene_086.txt',
    # 'scene_087': './results/scene_087.txt',
    # 'scene_088': './results/scene_088.txt',
    # 'scene_089': './results/scene_089.txt',
    # 'scene_090': './results/scene_090.txt',
}

cam_ids = {
    # AIHub
    "scene_042": sorted(
        [
            int(cam.split("_")[-1])
            for cam in os.listdir("videos/val/scene_042")
            if cam.startswith("camera_")
        ]
    ),
    # Val
    # 'scene_041': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_041') if cam.startswith('camera_')]),
    # 'scene_042': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_042') if cam.startswith('camera_')]),
    # 'scene_043': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_043') if cam.startswith('camera_')]),
    # 'scene_044': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_044') if cam.startswith('camera_')]),
    # 'scene_045': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_045') if cam.startswith('camera_')]),
    # 'scene_046': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_046') if cam.startswith('camera_')]),
    # 'scene_047': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_047') if cam.startswith('camera_')]),
    # 'scene_048': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_048') if cam.startswith('camera_')]),
    # 'scene_049': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_049') if cam.startswith('camera_')]),
    # 'scene_050': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_050') if cam.startswith('camera_')]),
    # 'scene_051': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_051') if cam.startswith('camera_')]),
    # 'scene_052': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_052') if cam.startswith('camera_')]),
    # 'scene_053': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_053') if cam.startswith('camera_')]),
    # 'scene_054': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_054') if cam.startswith('camera_')]),
    # 'scene_055': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_055') if cam.startswith('camera_')]),
    # 'scene_056': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_056') if cam.startswith('camera_')]),
    # 'scene_057': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_057') if cam.startswith('camera_')]),
    # 'scene_058': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_058') if cam.startswith('camera_')]),
    # 'scene_059': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_059') if cam.startswith('camera_')]),
    # 'scene_060': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/val/scene_060') if cam.startswith('camera_')]),
    # Test
    # 'scene_061': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_061') if cam.startswith('camera_')]),
    # 'scene_062': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_062') if cam.startswith('camera_')]),
    # 'scene_063': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_063') if cam.startswith('camera_')]),
    # 'scene_064': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_064') if cam.startswith('camera_')]),
    # 'scene_065': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_065') if cam.startswith('camera_')]),
    # 'scene_066': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_066') if cam.startswith('camera_')]),
    # 'scene_067': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_067') if cam.startswith('camera_')]),
    # 'scene_068': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_068') if cam.startswith('camera_')]),
    # 'scene_069': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_069') if cam.startswith('camera_')]),
    # 'scene_070': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_070') if cam.startswith('camera_')]),
    # 'scene_071': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_071') if cam.startswith('camera_')]),
    # 'scene_072': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_072') if cam.startswith('camera_')]),
    # 'scene_073': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_073') if cam.startswith('camera_')]),
    # 'scene_074': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_074') if cam.startswith('camera_')]),
    # 'scene_075': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_075') if cam.startswith('camera_')]),
    # 'scene_076': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_076') if cam.startswith('camera_')]),
    # 'scene_077': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_077') if cam.startswith('camera_')]),
    # 'scene_078': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_078') if cam.startswith('camera_')]),
    # 'scene_079': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_079') if cam.startswith('camera_')]),
    # 'scene_080': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_080') if cam.startswith('camera_')]),
    # 'scene_081': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_081') if cam.startswith('camera_')]),
    # 'scene_082': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_082') if cam.startswith('camera_')]),
    # 'scene_083': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_083') if cam.startswith('camera_')]),
    # 'scene_084': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_084') if cam.startswith('camera_')]),
    # 'scene_085': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_085') if cam.startswith('camera_')]),
    # 'scene_086': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_086') if cam.startswith('camera_')]),
    # 'scene_087': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_087') if cam.startswith('camera_')]),
    # 'scene_088': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_088') if cam.startswith('camera_')]),
    # 'scene_089': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_089') if cam.startswith('camera_')]),
    # 'scene_090': sorted([int(cam.split('_')[-1]) for cam in os.listdir('/workspace/videos/test/scene_090') if cam.startswith('camera_')]),
}


def get_reader_writer(source):
    src_paths = sorted(
        os.listdir(source), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    src_paths = [os.path.join(source, s) for s in src_paths]

    fps = 1
    wi, he = 1920, 1080
    os.makedirs("output_videos/" + source.split("/")[-2], exist_ok=True)
    # dst = 'output_videos/' + source.replace('/','').replace('.','') + '.mp4'
    dst = (
        f"output_videos/{source.split('/')[-2]}/"
        + source.split("/")[-3]
        + "_"
        + source.split("/")[-2]
        + source.split("/")[-1]
        + ".mp4"
    )
    video_writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wi, he))
    print(f"{source}'s total frames: {len(src_paths)}")

    return [src_paths, video_writer]


def get_map_writer(source):
    fps = 1
    wi, he = 1920, 1080 * 2
    os.makedirs("output_videos/" + source.split("/")[-2], exist_ok=True)
    dst = (
        f"output_videos/{source.split('/')[-2]}/"
        + source.split("/")[-3]
        + "_"
        + source.split("/")[-2]
        + "map"
        + ".mp4"
    )
    video_writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wi, he))
    return video_writer


def get_result_writer(source):
    fps = 1
    wi, he = 1920 * 2, 1080 * 2
    os.makedirs("output_videos/" + source.split("/")[-2], exist_ok=True)
    dst = (
        f"output_videos/{source.split('/')[-2]}/"
        + source.split("/")[-3]
        + "_"
        + source.split("/")[-2]
        + "result"
        + ".mp4"
    )
    video_writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wi, he))
    return video_writer


def finalize_cams(src_handlers, result_writer, map_writer):
    for s, w in src_handlers:
        w.release()
        print(f"{w} released")

    result_writer.release()
    print(f"{result_writer} released")

    map_writer.release()
    print(f"{map_writer} released")


def write_vids(
    trackers,
    imgs,
    src_handlers,
    pose,
    colors,
    mc_tracker,
    cur_frame=0,
    result_writer=None,
    write_result=False,
    map_writer=None,
    map_image=None,
    write_map=False,
    track_records=None,
):
    writers = [w for s, w in src_handlers]
    result_imgs = [] if write_result else None
    gid_2_lenfeats = {}
    sc_map_image = map_image.copy()
    mc_map_image = map_image.copy()

    for track in mc_tracker.tracked_mtracks + mc_tracker.lost_mtracks:
        if track.is_activated:
            gid_2_lenfeats[track.track_id] = len(track.features)
        else:
            gid_2_lenfeats[-2] = len(track.features)

    for tracker_index, (tracker, img, w) in enumerate(zip(trackers, imgs, writers)):
        outputs = [
            t.tlbr.tolist()
            + [t.score, t.global_id, gid_2_lenfeats.get(t.global_id, -1)]
            for t in tracker.tracked_stracks
        ]
        pose_result = [t.pose for t in tracker.tracked_stracks if t.pose is not None]
        if True:
            img = visualize(outputs, img, colors, pose, pose_result, cur_frame)
            w.write(img)
            if write_result:
                result_imgs.append(img)

        if write_map:
            is_mc = True
            tracks = tracker.tracked_stracks
            for t in tracks:
                location = tuple(map(int, t.location[0]))
                track_id = t.global_id
                color = (colors[track_id % 80] * 255).astype(np.uint8).tolist()
                cv2.putText(
                    sc_map_image,
                    f"{str(track_id)}[{tracker_index}]",
                    location,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
                cv2.circle(sc_map_image, location, 10, color, -1)
                
    track_records = create_track_records(trackers)

    if write_map:
        is_mc = True
        tracks = mc_tracker.tracked_mtracks

        for t in tracks:
            location = t.curr_coords[0].astype(int).tolist()
            track_id = t.global_id
            color = (colors[track_id % 80] * 255).astype(np.uint8).tolist()
            cv2.putText(
                mc_map_image,
                str(track_id),
                location,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            cv2.circle(mc_map_image, location, 10, color, -1)

    # draw camera track using mc_tracker
    if False:
        copied_imgs = [img.copy() for img in imgs]
        m = 2
        for mtrack in mc_tracker.tracked_mtracks:
            img_paths = mtrack.img_paths
            path_tlbr = mtrack.path_tlbr
            track_id = mtrack.global_id
            text = f"{track_id}"
            txt_color = (
                (0, 0, 0) if np.mean(colors[track_id % 80]) > 0.5 else (255, 255, 255)
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4 * m, 1 * m)[0]
            txt_bk_color = (colors[track_id % 80] * 255 * 0.7).astype(np.uint8).tolist()
            color = (colors[track_id % 80] * 255).astype(np.uint8).tolist()
            for img_path in img_paths:
                tlbr = path_tlbr[img_path].astype(int)
                x0, y0 = tlbr[:2]
                img = copied_imgs[img_path]
                cv2.rectangle(img, tlbr[:2], tlbr[2:], color, 3)
                cv2.rectangle(
                    img,
                    (x0, y0 - 2),
                    (x0 + txt_size[0] + 1, y0 - int(1.5 * txt_size[1])),
                    txt_bk_color,
                    -1,
                )
                cv2.putText(
                    img,
                    text,
                    (x0, int(y0 - txt_size[1] / 2)),
                    font,
                    0.4 * m,
                    txt_color,
                    thickness=1 * m,
                )
        result_imgs = copied_imgs

    if write_map:
        map_image = map_image[-1080:, :]

    # write table
    if True:
        def generate_track_records(track_records):
            track_records = pd.DataFrame(track_records).T
            # 필터링: track_id가 음수가 아닌 데이터만 선택
            filtered_data = track_records[track_records['track_id'] > 0]
            # 정렬: duration을 기준으로 오름차순
            sorted_data = filtered_data.sort_values(by='duration', ascending=True)
            # 상위 7개의 데이터 선택
            top_data = sorted_data.head(7)
            # 5개 컬럼 선택
            result = top_data[["track_id", "X", "Y", "distance", "duration"]]
            return result

        def generate_random_table(rows, cols):
            data = {
                f"Col {j+1}": [random.randint(1, 100) for _ in range(rows)]
                for j in range(cols)
            }
            return pd.DataFrame(data)

        # OpenCV 화면에 표를 그리는 함수
        def draw_table(
            image, table, start_x, start_y, cell_width, cell_height, font_scale=0.6
        ):
            """
            Draws a Pandas DataFrame table on an OpenCV image.

            Args:
                image (numpy.ndarray): OpenCV image to draw on.
                table (pandas.DataFrame): Table to draw.
                start_x (int): X-coordinate to start drawing.
                start_y (int): Y-coordinate to start drawing.
                cell_width (int): Width of each cell.
                cell_height (int): Height of each cell.
                font_scale (float): Font size for text.
            """
            # Table dimensions
            rows, cols = table.shape
            end_x = start_x + cols * cell_width
            end_y = start_y + (rows + 1) * cell_height

            # Draw horizontal lines
            for i in range(rows + 2):
                y = start_y + i * cell_height
                cv2.line(image, (start_x, y), (end_x, y), (255, 255, 255), 10)

            # Draw vertical lines
            for j in range(cols + 1):
                x = start_x + j * cell_width
                cv2.line(image, (x, start_y), (x, end_y), (255, 255, 255), 5)

            # Add text in cells
            columns_names = ["Track ID", "X", "Y", "Distance", "Duration"]
            for j, column_name in enumerate(columns_names):
                text_size = cv2.getTextSize(
                    column_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )[0]
                text_x = start_x + j * cell_width + (cell_width - text_size[0]) // 2
                text_y = start_y + (cell_height + text_size[1]) // 2
                cv2.putText(
                    image,
                    column_name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    10,
                )

            for i in range(rows):
                for j in range(cols):
                    text = str(table.iloc[i, j])
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )[0]
                    text_x = start_x + j * cell_width + (cell_width - text_size[0]) // 2
                    text_y = (
                        start_y
                        + (i + 1) * cell_height
                        + (cell_height + text_size[1]) // 2
                    )
                    cv2.putText(
                        image,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        10,
                    )

        # 표 데이터를 랜덤으로 생성하는 함수
        table_frame = np.zeros_like(map_image, dtype=np.uint8)

        # Generate a random table
        rows, cols = 7, 5
        random_table = generate_random_table(rows, cols)
        record_table = generate_track_records(track_records)

        # Draw the table in the bottom-left corner
        draw_table(
            table_frame,
            record_table,
            start_x=210,
            start_y=80,
            cell_width=300,
            cell_height=120,
            font_scale=1.8,
        )

    is_table = False
    window = table_frame if is_table else sc_map_image
    stacked_map = np.vstack([mc_map_image, window])

    if write_result:
        stacked_img = np.vstack(result_imgs)
        result_stack = np.hstack([stacked_img, stacked_map])
        result_writer.write(result_stack)

    return result_stack


def write_results_testset(result_lists, result_path):
    dst_folder = str(Path(result_path).parent)
    os.makedirs(dst_folder, exist_ok=True)
    # write multicam results
    with open(result_path, "w") as f:
        print(result_path)
        for result in result_lists:
            for r in result:
                t, l, w, h = r["tlwh"]
                xworld, yworld = r["2d_coord"]
                row = [
                    r["cam_id"],
                    r["track_id"],
                    r["frame_id"],
                    int(t),
                    int(l),
                    int(w),
                    int(h),
                    float(xworld),
                    float(yworld),
                ]
                row = " ".join([str(r) for r in row]) + "\n"
                # row = " ".join(row)
                f.write(row)


def update_result_lists_testset(trackers, result_lists, frame_id, cam_ids, scene):
    results_frame = [[] for i in range(len(result_lists))]
    results_frame_feat = []
    for tracker, result_frame, result_list, cam_id in zip(
        trackers, results_frame, result_lists, cam_ids
    ):
        for track in tracker.tracked_stracks:
            if track.global_id < 0:
                continue
            result = {
                "cam_id": int(cam_id),
                "frame_id": frame_id,
                "track_id": track.global_id,
                "sct_track_id": track.track_id,
                "tlwh": list(map(lambda x: int(x), track.tlwh.tolist())),
                "2d_coord": track.location[0].tolist(),
            }
            result_ = list(result.values())
            result_list.append(result)


def visualize(dets, img, colors, pose, pose_result, cur_frame):
    m = 2
    if len(dets) == 0:
        return img

    keypoints = [p["keypoints"][:, :2] for p in pose_result]
    scores = [p["keypoints"][:, 2] for p in pose_result]
    img = visualize_kpt(img, keypoints, scores, thr=0.3)

    for obj in dets:
        score = obj[4]
        track_id = int(obj[5])
        len_feats = " " if obj[6] == 50 else obj[6]
        x0, y0, x1, y1 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

        color = (colors[track_id % 80] * 255).astype(np.uint8).tolist()
        text = "{} : {:.1f}% | {}".format(track_id, score * 100, len_feats)
        txt_color = (
            (0, 0, 0) if np.mean(colors[track_id % 80]) > 0.5 else (255, 255, 255)
        )
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4 * m, 1 * m)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (colors[track_id % 80] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - 1),
            (x0 + txt_size[0] + 1, y0 - int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            img, text, (x0, y0 - txt_size[1]), font, 0.4 * m, txt_color, thickness=1 * m
        )

    return img


def visualize_kpt(img, keypoints, scores, thr=0.3) -> np.ndarray:

    skeleton = [
        [12, 13],
        [13, 0],
        [13, 1],
        [0, 1],
        [6, 7],
        [0, 2],
        [2, 4],
        [1, 3],
        [3, 5],
        [0, 6],
        [1, 7],
        [6, 8],
        [8, 10],
        [7, 9],
        [9, 11],
    ]
    palette = [
        [51, 153, 255],
        [0, 255, 0],
        [255, 128, 0],
        [255, 255, 255],
        [255, 153, 255],
        [102, 178, 255],
        [255, 51, 51],
    ]
    link_color = [3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
    point_color = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]

    # draw keypoints and skeleton
    for kpts, score in zip(keypoints, scores):
        for kpt, color in zip(kpts, point_color):
            cv2.circle(
                img, tuple(kpt.astype(np.int32)), 2, palette[color], 2, cv2.LINE_AA
            )
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(
                    img,
                    tuple(kpts[u].astype(np.int32)),
                    tuple(kpts[v].astype(np.int32)),
                    palette[color],
                    1,
                    cv2.LINE_AA,
                )

    return img
