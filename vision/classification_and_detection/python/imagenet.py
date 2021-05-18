"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import re
import time

import cv2
import numpy as np

import dataset

import io
import requests
import cli_colors
import socket

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")


class Imagenet(dataset.Dataset):

    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
                 image_format="NHWC", pre_process=None, count=None, cache_dir=None):
        super(Imagenet, self).__init__()
        if image_size is None:
            self.image_size = [224, 224, 3]
        else:
            self.image_size = image_size
        if not cache_dir:
            cache_dir = os.getcwd()
        self.image_list = []
        self.label_list = []
        self.count = count
        self.use_cache = use_cache
        self.cache_dir = os.path.join(
            cache_dir, "preprocessed", name, image_format)
        self.data_path = data_path
        self.pre_process = pre_process
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False

        not_found = 0
        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "val_map.txt")

        os.makedirs(self.cache_dir, exist_ok=True)

        start = time.time()
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(data_path, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                os.makedirs(os.path.dirname(os.path.join(
                    self.cache_dir, image_name)), exist_ok=True)
                dst = os.path.join(self.cache_dir, image_name)
                if not os.path.exists(dst + ".npy"):
                    cli_colors.color_print(
                        f"Preprocessing {dst}", cli_colors.RED)
                    # cache a preprocessed version of the image
                    # TODO: make this multi threaded ?
                    img_org = cv2.imread(src)
                    processed = self.pre_process(
                        img_org, need_transpose=self.need_transpose, dims=self.image_size)
                    np.save(dst, processed)

                self.image_list.append(image_name)
                self.label_list.append(int(label))

                # limit the dataset if requested
                if self.count and len(self.image_list) >= self.count:
                    break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

        self.label_list = np.array(self.label_list)
        # self.listen_to_data()

    def listen_to_data(self):
        self.sockfd = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.sockfd.bind(("0.0.0.0", 8085))
        print(f"Waiting")
        self.sockfd.listen()
        self.clifd, addr = self.sockfd.accept()
        self.clifd.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        print(self.clifd.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        print(f"connected to {addr}")

    def listen_to_data_udp(self):
        self.sockfd = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sockfd.bind(("0.0.0.0", 8085))
        print(f"Waiting")
        # self.sockfd.listen()
        # self.clifd, addr = self.sockfd.accept()
        self.sockfd.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        print(self.sockfd.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        # print(f"connected to {addr}")

    url = "http://192.168.39.55:8082/"
    # url = "http://127.0.0.1:8082/"

    def get_item_remote(self, nr):
        """Get image by number in the list."""
        path = f"{self.url}{self.image_list[nr]}.npy"
        # cli_colors.color_print(f"Fetching {path}", cli_colors.YELLOW)
        img = requests.get(path).content
        imgio = io.BytesIO(img)
        img = np.load(imgio)
        # dst = os.path.join(self.cache_dir, self.image_list[nr])
        # img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    def get_item_pipe(self, nr):
        print("reading pipe")
        data = self.clifd.recv(602240, socket.MSG_WAITALL)
        if not data:
            print("Error in connection")
        img = np.load(io.BytesIO(data))
        return img, self.label_list[nr]

    def get_item_pipe_udp(self, nr):
        print("reading pipe")
        data: bytes = b''
        fragments = {}
        while len(fragments) < 12:
            buf, addr = self.sockfd.recvfrom(2**10 * 50 + 1)
            index, buf = int(buf[0]), buf[1:]
            print(index)
            fragments[index] = buf
            data += buf
        for i in range(12):
            data+=fragments[i]
        len(data)            
        img = np.load(io.BytesIO(data))
        return img, self.label_list[nr]

    def get_item_org(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    get_item = get_item_org

    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src
