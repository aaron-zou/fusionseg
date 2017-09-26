#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import argparse
import fnmatch
import numpy as np
import tables

# Caffe binary location from deeplab-v2 installation
caffe_binary = ('/vision/vision_users/azou/deeplab-public-ver2/distribute/'
                'bin/caffe.bin')

# DenseCRF binary location (Efficient Inference in Fully Connected CRFs
crf_binary = ('/vision/vision_users/azou/lib/densecrf/build/examples/'
              'dense_inference')


def get_data_dir(model_type):
    """Expect specific folder to use for data source."""
    base_dir = os.getcwd()
    if model_type == 'appearance':
        image_dir = os.path.join(base_dir, 'images')
        if not os.path.isdir(image_dir):
            sys.exit("Image folder not found")
        return image_dir
    elif model_type == 'motion':
        image_dir = os.path.join(base_dir, 'motion_images')
        if not os.path.isdir(image_dir):
            sys.exit("Need optical flow images stored in motion_images/")
        return image_dir
    else:
        sys.exit('Internal error')


def run_model(model_type, ext):
    """Run pre-trained model for a single stream and save output as MATLAB blob.

    """
    base_dir = os.getcwd()
    image_dir = get_data_dir(model_type)
    model_dir = os.path.join(base_dir, model_type) + '/'

    # Find matching images of correct type
    image_list = fnmatch.filter(os.listdir(image_dir), '*.' + ext)
    image_list.sort()

    input_list_file = "{}/{}_image_list.txt".format(base_dir, model_type)
    output_list_file = "{}/{}_output_list.txt".format(base_dir, model_type)

    # Parse filenames of files
    with open(input_list_file, 'w') as input_list, \
            open(output_list_file, 'w') as output_list:
        for img in image_list:
            input_list.write('/' + img + '\n')
            prefix = img.split('.')[0]
            output_list.write(prefix + '\n')

    # Process template prototxt
    template_file = open(model_dir + model_type +
                         '_stream_template.prototxt').readlines()

    test_file_path = model_dir + model_type + '_stream.prototxt'
    test_file = open(test_file_path, 'w')

    tokens = {}
    tokens['${IMAGE_DIR}'] = "root_folder: \"{}\"".format(image_dir)
    tokens['${OUTPUT_DIR}'] = "prefix: \"{}\"".format(image_dir)

    tokens['${IMAGE_LIST}'] = "source: \"{}\"".format(input_list_file)
    tokens['${IMAGE_OUTPUT_LIST}'] = "source: \"{}\"".format(output_list_file)

    for line in template_file:
        line = line.rstrip()
        for key in tokens:
            if line.find(key) != -1:
                line = '\t' + tokens[key]
                break
        test_file.write(line + '\n')
    test_file.close()

    # Run Caffe binary
    weight_file_path = model_dir + model_type + '_stream.caffemodel'
    cmd = caffe_binary + ' test --model=' + test_file_path + ' --weights=' + \
        weight_file_path + ' --gpu=all --iterations=' + str(len(image_list))
    print(cmd)
    os.system(cmd)


def crf_process(model_type, ext):
    """Run CRF inference on original images and segmentation annotations.

    Saves new segmentation images.
    """
    # Iterate through each image (get path to image and annotation)
    base_dir = os.getcwd()
    image_dir = get_data_dir(model_type)
    output_list_file = "{}/{}_output_list.txt".format(base_dir, model_type)
    with open(output_list_file, 'r') as f:
        prefixes = f.read().splitlines()

    for prefix in prefixes:
        img_path = os.path.join(image_dir, "{}.{}".format(prefix, ext))
        output_path = os.path.join(image_dir, "{}_seg.{}".format(prefix, ext))

        # Run DenseCRF binary and generate output files
        cmd = "{} {} {} {}".format(
            crf_binary, img_path, anno_path, output_path)
        print(cmd)
        os.system(cmd)


def make_parser():
    """Create a simple argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type",
        choices=['appearance', 'motion'], help="Which stream to process.")
    parser.add_argument_group()
    parser.add_argument("--run_model", action="store_true",
                        help="Run pretrained model.")
    parser.add_argument("ext", help="File extension of images.")
    parser.add_argument("--crf", action="store_true",
                        help="Run CRF postprocessing stage.")
    return parser.parse_args()


def main():
    args = make_parser()
    if args.run_model:
        run_model(args.type, args.ext)
    if args.crf:
        crf_process(args.type)


if __name__ == '__main__':
    main()
