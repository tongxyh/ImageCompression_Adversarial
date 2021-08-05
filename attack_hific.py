# Copyright 2021 Tong Chen All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
attack hific models.
"""

import argparse
import collections
import glob
import itertools
import os
import sys

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from hific import configs
from hific import helpers
from hific import model


# Show custom tf.logging calls.
tf.logging.set_verbosity(tf.logging.INFO)

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  return tf.expand_dims(image, 0)


def write_png(filename, image):
  """Writes a PNG image file."""
  image = tf.squeeze(image, 0)
  if image.dtype.is_floating:
    image = tf.round(image)
  if image.dtype != tf.uint8:
    image = tf.saturate_cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)

def gradient_net(): 
    sobel_x = np.array([[-0.25, -0.2 ,  0.  ,  0.2 ,  0.25],
                      [-0.4 , -0.5 ,  0.  ,  0.5 ,  0.4 ],
                      [-0.5 , -1.  ,  0.  ,  1.  ,  0.5 ],
                      [-0.4 , -0.5 ,  0.  ,  0.5 ,  0.4 ],
                      [-0.25, -0.2 ,  0.  ,  0.2 ,  0.25]])

    sobel_y = np.array([[-0.25, -0.4 , -0.5 , -0.4 , -0.25],
                      [-0.2 , -0.5 , -1.  , -0.5 , -0.2 ],
                      [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                      [ 0.2 ,  0.5 ,  1.  ,  0.5 ,  0.2 ],
                      [ 0.25,  0.4 ,  0.5 ,  0.4 ,  0.25]])

    filters = np.concatenate([[sobel_x], [sobel_y]])  # shape: (2, 5, 5)
    filters = np.expand_dims(filters, -1)  # shape: (2, 5, 5, 1)
    filters = filters.transpose(1, 2, 3, 0)  # shape: (5, 5, 1, 2)
    ans_x = tf.nn.conv2d((img / 255.0).astype('float32'),
                   filters[:,:,:,0:1],
                   strides=[1, 1, 1, 1],
                   padding='SAME')
    ans_y = tf.nn.conv2d((img / 255.0).astype('float32'),
                   filters[:,:,:,1:2],
                   strides=[1, 1, 1, 1],
                   padding='SAME')              
    return tf.math.tanh(tf.abs(ans_x) + tf.abs(ans_y))               

def attack_trained_model(args,
                       config_name,
                       ckpt_dir,
                       out_dir,
                       images_glob,
                       tfds_arguments: helpers.TFDSArguments,
                       max_images=None,
                       ):
  """Attack a trained model."""
  config = configs.get_config(config_name)
  hific = model.HiFiC(config, helpers.ModelMode.ATTACK)

  lamb = args.lamb_attack
  filename = "/ct/datasets/kodak/kodim19.png"
  print(f"[Input] {filename}")
  input_image = read_png(filename)
  with tf.Session() as sess:
      shape = sess.run(tf.shape(input_image))

  # noise = tf.random.normal(shape, mean=0, stddev=1)/10.0
  # with tf.Session() as sess:
  #     noise_ = sess.run(noise)
  print(shape)
  noise_ = np.random.randn(shape[0],shape[1],shape[2],shape[3]).astype(np.float32)*(255.0/10.0)
  with tf.name_scope("attacker") as scope:
    noise = tf.Variable(noise_, name="noise")
    mask = tf.Variable(tf.ones_like(noise), name="mask")
    noise_clipped = tf.clip_by_value(noise*mask, -50, 50.)
  print("[ATTACK] CLIP input to 0-255")
  input_attack = tf.clip_by_value(tf.math.add(input_image,noise_clipped), 0, 255.)
  output_image, bitstring = hific.build_model(input_attack)
  mse_in = tf.math.reduce_mean(tf.square(input_attack - input_image))

  writer = tf.summary.FileWriter("/ct/code/compression/models/tf_logs")

  # output_image, bitstring = hific.build_model(input_image)
  with tf.name_scope("attacker_opt") as scope:
    update_mask = tf.assign(mask, tf.math.tanh(tf.square(output_image-input_image) / (tf.square(noise)+0.0001)))
    loss_i = tf.math.reduce_mean(tf.square(noise))
    loss_o = tf.math.reduce_mean(tf.square(output_image-input_image))
    cost = loss_i + lamb * (255*255-loss_o)
    print("[TODO] !!! gradient mask")
    
    print("[TODO] !!! clip with gradient")
    print("[ATTACK] learning rate decay x0.9 every 1000 steps")
    global_step = tf.Variable(0, trainable=False)
    initial_learning_rate = 1.0
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                             global_step=global_step,
                                             decay_steps=3000,decay_rate=0.95)
    # TODO: optimize only noise
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=[noise], global_step=global_step)
  
  # g = tf.gradients([output_image], [input_image])
  input_image = tf.cast(tf.round(input_attack[0, ...]), tf.uint8)
  output_image = tf.cast(tf.round(output_image[0, ...]), tf.uint8)

  os.makedirs(out_dir, exist_ok=True)

  accumulated_metrics = collections.defaultdict(list)

  with tf.Session() as sess:  
    
    # hific.restore_trained_model(sess, ckpt_dir) 
    sess.run(tf.variables_initializer(tf.global_variables()))
    variables = tf.global_variables()
    variables_to_restore = [v for v in variables if v.name.split('/')[0] not in ['attacker', 'attacker_opt']]
    saver = tf.train.Saver(variables_to_restore)
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, latest_ckpt)

    hific.prepare_for_arithmetic_coding(sess)
    steps = 100001
    for i in range(steps):
      if max_images and i == max_images:
        break
      try:
        opt_v, inp_np, otp_np, mse_in_np, ls_in, ls_out, loss, bitstring_np, _, step_v = \
          sess.run([optimizer, input_image, output_image, mse_in, loss_i, loss_o, cost, bitstring, update_mask, global_step])
        
        print(f"step: {step_v},\tLoss_all: {loss:.4f}, Loss_in: {mse_in_np:.4f}, Loss_out: {ls_out:.4f}")
        
        # print(noise_v.min(), noise_v.max())  
        
        h, w, c = otp_np.shape
        assert c == 3
        bpp = get_arithmetic_coding_bpp(
            bitstring, bitstring_np, num_pixels=h * w)

        # metrics = {'psnr': get_psnr(inp_np, otp_np),
        #            'bpp_real': bpp}

        # metrics_str = ' / '.join(f'{metric}: {value:.5f}'
        #                          for metric, value in metrics.items())
        # print(f'Image {i: 4d}: {metrics_str}, saving in {out_dir}...')

        # for metric, value in metrics.items():
        #   accumulated_metrics[metric].append(value)

        # # Save images.
        if i % 1000 == 0:
          name = 'test'
          Image.fromarray(inp_np).save(
              os.path.join(out_dir, f'{name}_input_{i}_{ls_in:.4f}_{ls_out:.4f}.png'))
          Image.fromarray(otp_np).save(
              os.path.join(out_dir, f'{name}_outpt_{i}_{bpp:.3f}_{ls_in:.4f}_{ls_out:.4f}.png'))
          if mse_in_np > 0.03*255.*255. or ls_out > 0.04*255*255:
            print(f"[Done!] PSNR_in: {-10*np.log10(mse_in_np/255./255.)} PSNR_out: {-10*np.log10(ls_out/255./255.)}")
            break
      except tf.errors.OutOfRangeError:
        print('No more inputs.')
        break
    writer = tf.summary.FileWriter(logdir="/ct/code/compression/models/tf_logs", graph=sess.graph)
    writer.flush()
  print('\n'.join(f'{metric}: {np.mean(values)}'
                  for metric, values in accumulated_metrics.items()))
  print('Done!')


def get_arithmetic_coding_bpp(bitstring, bitstring_np, num_pixels):
  """Calculate bitrate we obtain with arithmetic coding."""
  # TODO(fab-jul): Add `compress` and `decompress` methods.
  packed = tfc.PackedTensors()
  packed.pack(tensors=bitstring, arrays=bitstring_np)
  return len(packed.string) * 8 / num_pixels


def get_psnr(inp, otp):
  mse = np.mean(np.square(inp.astype(np.float32) - otp.astype(np.float32)))
  psnr = 20. * np.log10(255.) - 10. * np.log10(mse)
  return psnr


def get_image_names(images_glob):
  if not images_glob:
    return {}
  return {i: os.path.splitext(os.path.basename(p))[0]
          for i, p in enumerate(sorted(glob.glob(images_glob)))}


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', required=True,
                      choices=configs.valid_configs(),
                      help='The config to use.')
  parser.add_argument('--ckpt_dir', required=True,
                      help=('Path to the folder where checkpoints of the '
                            'trained model are.'))
  parser.add_argument('--out_dir', required=True, help='Where to save outputs.')

  parser.add_argument('--images_glob', help='If given, use TODO')
  parser.add_argument("-la", dest="lamb_attack", type=float, default=0.2, help="attack lambda")
  helpers.add_tfds_arguments(parser)

  args = parser.parse_args(argv[1:])
  return args


def main(args):
  attack_trained_model(args, args.config, args.ckpt_dir, args.out_dir,
                     args.images_glob,
                     helpers.parse_tfds_arguments(args))


if __name__ == '__main__':
  main(parse_args(sys.argv))