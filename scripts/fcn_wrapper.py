#!/usr/bin/env python3

"""
FCNWrapper

author : yu.omori
email  : yu.omori0521@gmail.com
"""

import configparser
import os
import re
import sys
import time

import chainer
import cv2
import numpy as np
import skimage.io

import fcn


class FCNWrapper:
    """FCN class"""

    def __init__(self, modelfilepath, n_class, gpu_id):
        """Initialization"""
        self.n_class = n_class
        self.modelfilepath = modelfilepath
        self.gpu_id = gpu_id
        self.model = None

    def load_model(self):
        """loading model file"""
        match = re.match('^fcn(32|16|8)s.*$', os.path.basename(self.modelfilepath))
        if match is None:
            err_msg = 'Unsupported model filename: %s' % self.modelfilepath
            raise NameError(err_msg)
        else:
            model_name = 'FCN%ss' % match.groups()[0]
            model_class = getattr(fcn.models, model_name)
            self.model = model_class(n_class=self.n_class)
            chainer.serializers.load_npz(self.modelfilepath, self.model)

        if self.gpu_id >= 0:
            chainer.cuda.get_device(self.gpu_id).use()
            self.model.to_gpu()

    @classmethod
    def load_img(cls, imgfilepath):
        """load img from imgfilepath"""
        if os.path.exists(imgfilepath):
            img = skimage.io.imread(imgfilepath)
        else:
            err_msg = "%s does not exist" % imgfilepath
            raise NameError(err_msg)
        return img

    def inference(self, img):
        """inference"""

        input_img, = fcn.datasets.transform_lsvrc2012_vgg16((img,))
        input_img = input_img[np.newaxis, :, :, :]
        if self.gpu_id >= 0:
            input_img = chainer.cuda.to_gpu(input_img)

        # forward
        with chainer.no_backprop_mode():
            input_img = chainer.Variable(input_img)
            with chainer.using_config('train', False):
                self.model(input_img)
                lbl_pred = chainer.functions.argmax(self.model.score, axis=1)[0]
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)

        # visualize
        viz = fcn.utils.visualize_segmentation(
            lbl_pred=lbl_pred, img=img, n_class=self.n_class,
            label_names=fcn.datasets.VOC2012ClassSeg.class_names)
        return viz

def read_config(configfilepath):
    """
    reading config and set parameter
    """

    config = configparser.ConfigParser()
    config.read(configfilepath)

    try:
        modelfilepath = config.get("COMMON", "modelfilepath")
        classnum = config.getint("COMMON", "classnum")
        gpu_id = config.getint("COMMON", "gpu_id")
        return modelfilepath, classnum, gpu_id

    except configparser.Error as config_parse_err:
        raise config_parse_err

def main():
    """main"""

    try:
        modelfilepath, n_class, gpu_id = read_config("./config/fcn_wrapper.ini")
    except configparser.Error as config_parse_err:
        print(config_parse_err)
        sys.exit(1)

    if not os.path.exists(modelfilepath):
        print("%s does not exist" % modelfilepath)
        sys.exit(1)

    fcn_predictor = FCNWrapper(modelfilepath, n_class, gpu_id)

    try:
        fcn_predictor.load_model()
    except NameError:
        sys.exit(1)

    paused = False
    delay = {True: 0, False: 1}
    cap = cv2.VideoCapture(0)

    actual_fps = 0

    while True:
        try:
            _, frame = cap.read()
        except cv2.Error:
            break
        start_time = time.time()
        pred_img = fcn_predictor.inference(frame)
        fcn_fps = 1.0 / (time.time() - start_time)
        cv2.putText(pred_img, 'UI FPS = %f, FCN FPS = %f'
                    % (actual_fps, fcn_fps), (20, 20), 0, 0.5, (0, 0, 255))
        cv2.imshow("FCN result", pred_img)
        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break
        actual_fps = 1.0 / (time.time() - start_time)

if __name__ == "__main__":
    main()
