import os
import matplotlib.pyplot as plt
import argparse
#from data import DIV2K
from model.srgan import generator#, discriminator
#from train import SrganTrainer, SrganGeneratorTrainer
from model import resolve_single
from utils import load_image
from PIL import Image
import tensorflow as tf
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('input_path')
parser.add_argument('save_path')
parser.add_argument('weight_path')
args = parser.parse_args()
input_path = args.input_path
gan_path = args.save_path
weight_path = args.weight_path
#usage python srgan_test.py input_file save_path weight_path

# Location of model weights (needed for demo)
#weights_file = lambda filename: os.path.join(weights_path, filename)

#os.makedirs(weights_dir, exist_ok=True)

#pre_generator = generator()
gan_generator = generator()

print(weight_path)
#pre_generator.load_weights(weights_file('pre_generator_01.h5'))
gan_generator.load_weights(weight_path)

def resolve_and_plot(lr_image_path):
    lr, size = load_image(lr_image_path)
    
    lr = cv2.cvtColor(lr, cv2.COLOR_RGBA2RGB)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    lr = cv2.GaussianBlur(lr, (7,7),0)

    #pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)
    
    #images = [lr, pre_sr, gan_sr]
    img_path = lr_image_path.split('.')
    
    save_image(gan_sr.numpy(), gan_path)
    #save_image(pre_sr.numpy(), 'demo/car1_pre_20.png')

def save_image(arr, path):
    image = Image.fromarray(arr)
    image.save(path)

resolve_and_plot(input_path)
