#!/usr/bin/env python3
# lanenet_predict.py

import sys
import os
# 현재 파일 위치 기준으로 lanenet_model 폴더 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__)))


import tensorflow as tf
import numpy as np
import cv2
import yaml

#from lanenet_model import lanenet  # GitHub lanenet-lane-detection의 lanenet_model 사용
from lanenet_inference.lanenet_model import lanenet
from lanenet_model.lanenet import LaneNet

#from config import global_config

from local_utils.config_utils import parse_config_utils
CFG = parse_config_utils.lanenet_cfg

# config 로딩 (CFG 변수 사용)
net = LaneNet(phase='test', cfg=CFG)



def predict_lane(image_input, weights_path):
    """
    이미지 1장을 받아서 LANENet으로 차선 추론을 수행하고, 결과를 리턴한다.
    :param image_path: 테스트 이미지 경로
    :param weights_path: checkpoint 파일명 (확장자 없는 경로, 예: './model/tusimple_lanenet/tusimple_lanenet.ckpt')
    :return: binary_seg_image, instance_seg_image (numpy 배열)
    """

    # TensorFlow 세션 설정
    tf.reset_default_graph()
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    #phase_tensor = tf.constant('test', dtype=tf.string) 불필요, 원래는 binary_set_ret에서 쓰여야하는데 lanenet으로 받고 있어서 불필요

    # LANENet 모델 로드
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    #binary_seg_ret, instance_seg_ret = net.inference(input_tensor, phase_tensor) #TypeError: VariableScope: name_or_scope must be a string or VariableScope.---> 이 부분에서 name 파라미터가 None 이거나 문자열이 아닌 다른 형식으로 들어갔다는 뜻
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet') #즉, name='LaneNet'을 명시적으로 넣어주면 됨
    #이건 lanenet.py 안의 inference() 함수 정의가 이렇게 되어 있기 때문
    #def inference(self, input_tensor, name, reuse=None):   --->→ name은 필수 positional argument임

 
    # Saver 준비
    saver = tf.train.Saver()

    
    
    image = image_input.copy() 
    # 이미지 로드 & 전처리
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR) #TypeError: bad argument type for built-in operation 발생
                                                     #image_path는 cv2.imread()가 문자열 경로를 기대하는데, 현재 나는 실시간으로 grab_screen()을 통해 얻은 이미지 배열을 넘기고 있음. 즉, image_path가 **파일 경로(str)**가 아니라 **이미지(np.ndarray)**인 상태
    #함수의 첫 번째 인자는 파일 경로가 아니라 이미지 배열이어야 함. 그래서 아래줄로 교체
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # # 이미지가 이미 np.ndarray 형태로 들어옴 → cv2.imread 제거
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR) #drive파일과 중복리사이즈
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가



    # GPU 메모리 설정 추가
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True


    with tf.Session() as sess:
        # weight 로드
        #saver.restore(sess, weights_path)
        
        #weight_path에 파일이 없을시 예외처리
        try:
            saver.restore(sess, weights_path)
        except Exception as e:
            print(f"[ERROR] Failed to load Lanenet weights: {e}")
            return None, None

        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: image}
        )

        binary_seg_image = binary_seg_image[0]  # (256, 512, 1)
        instance_seg_image = instance_seg_image[0]  # (256, 512, 4)

        # 결과 후처리 (binary_seg_image를 0~255로 변환)
        binary_seg_image = (binary_seg_image * 255).astype(np.uint8)

        return binary_seg_image, instance_seg_image




#gta5에 혼돈을 줄이기 위해서 main함수 주석처리
""" 
def main():
    # 테스트 예제
    #test_image = './data/test_image.jpg'
    #weights_path = './model/tusimple_lanenet/tusimple_lanenet.ckpt'
    #weights_path = 'E:\gta5_project\AI_GTA5\lanenet_inference\lanenet_maybeshewill\tusimple_lanenet.ckpt'
    weights_path = "E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt"
    binary_mask, instance_mask = predict_lane(image_input, weights_path)

    # 결과 저장
    #cv2.imwrite('./output/binary_mask.png', binary_mask)

    # 결과 시각화 (옵션)
    cv2.imshow('binary_mask', binary_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""


""" 

if __name__ == '__main__':
    main()
"""