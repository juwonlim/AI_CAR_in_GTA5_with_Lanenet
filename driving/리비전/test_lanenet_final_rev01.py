#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""

import argparse
import os.path as ops
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append("E:/gta5_project/AI_GTA5/lanenet_inference")


print("[DEBUG] sys.path:", sys.path)
print("[DEBUG] Current dir:", os.getcwd())
print("[DEBUG] lanenet_model exists:", os.path.exists('../lanenet_model'))




import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')



#여기서부터 내가 만든 외부모듈 로딩
from data_collection.preprocess_for_lanenet import grab_screen


image_path = grab_screen
screen = cv2.resize(image_path, (1280, 720)) #1280x720으로 입력이 되야 잘됨




def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


#이미지의 대비를 향상 시키는 함수
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced



def test_lanenet(image_path, weights_path, with_lane_fit=True):
    """

    :param image_path:
    :param weights_path:
    :param with_lane_fit:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #image = enhance_contrast(image)  # 여기에 이미지 대비향상 추가
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0

    #이미지가 올바르게 읽혔는지 확인하는 코드
    print("[DEBUG] model loaded, image shape:", image.shape)  # <= 이 줄 추가


    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        #print("[DEBUG] Trying to restore from:", save_path)
        print("[DEBUG] Trying to restore from:", weights_path)

        saver.restore(sess=sess, save_path=weights_path)
        print("[DEBUG] Model checkpoint restored successfully.") # saver.restore 바로 아래 (model 복원 직후)

        t_start = time.time()
        loop_times = 500
        for i in range(loop_times):
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
        #sess.run직후 디버그 코드
        print("[DEBUG] sess.run 실행 완료")
        print("[DEBUG] binary_seg_image sum:", np.sum(binary_seg_image))
        print("[DEBUG] instance_seg_image shape:", instance_seg_image.shape)
        t_cost = time.time() - t_start
        t_cost /= loop_times
        LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=with_lane_fit,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']
        lane_params = postprocess_result['fit_params']

        if with_lane_fit:
            #lane_params is None 방어 코드 추가, TypeError: object of type 'NoneType' has no len() --> 이 에러 방지위해서
            if lane_params is None:
                print("[ERROR] No lane detected. Skipping this image.")
                return
        
            print("[DEBUG] lane_params:", lane_params) #결과 디버깅용 출력 추가
            
            LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
            
            for i in range(len(lane_params)):
                LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')

#--with_lane_fit false 설정으로 실행했을 때,
#embedding_feats → DBSCAN 클러스터링 시도함.
#그런데 embedding_feats가 비어 있음 (shape=(0, 4)), 즉 차선으로 인식된 포인트가 아예 없음.
#mask_image = None 상태가 되어 이후 시각화 시 mask_image[:, :, (2,1,0)] 에서 TypeError: 'NoneType' object is not subscriptable 발생


        #위의 이유로 방어코드 추가
        if mask_image is None:
            print("[WARNING] Lane not detected. Skipping visualization.")
            return

        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

    sess.close()
    return




if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path, args.weights_path, with_lane_fit=args.with_lane_fit)
