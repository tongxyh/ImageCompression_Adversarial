import os
import numpy as np
import json
# import mmd
from PIL import Image
from msssim import MultiScaleSSIM

def evaluate(submission_images, target_images, settings={}, logger=None):
	"""
	Calculates metrics for the given images.
	"""

	if settings is None:
		settings = {}
	if isinstance(settings, str):
		try:
			settings = json.loads(settings)
		except json.JSONDecodeError:
			settings = {}

	metrics = settings.get('metrics', ['PSNR', 'MSSSIM'])
	patch_size = settings.get('patch_size', 256)

	num_dims = 0
	sqerror_values = []
	msssim_values = []

	target_patches = []
	submission_patches = []
	rs = np.random.RandomState(0)

	for name in target_images:
		image0 = np.asarray(Image.open(target_images[name]).convert('RGB'), dtype=np.float32)
		image1 = np.asarray(Image.open(submission_images[name]).convert('RGB'), dtype=np.float32)
		# num_dims += image0.size
		num_dims += 1
  
		if 'PSNR' in metrics:
      		# sqerror_values.append(mse(image1, image0))
			sqerror_values.append(mse2psnr(mse(image1, image0)/image0.size))
   
		if 'MSSSIM' in metrics:
			# value = msssim(image0, image1) * image0.size
			value = msssim(image0, image1)
			if np.isnan(value):
				value = 0.0
				if logger:
					logger.warning('Evaluation of MSSSIM for `{name}` returned NaN. Assuming MSSSIM is zero.')
			msssim_values.append(value)
		# if 'KID' in metrics or 'FID' in metrics:
		# 	if image0.shape[0] >= patch_size and image0.shape[1] >= patch_size:
		# 		# extract random patches for later use
		# 		i = rs.randint(image0.shape[0] - patch_size + 1)
		# 		j = rs.randint(image0.shape[1] - patch_size + 1)
		# 		target_patches.append(image0[i:i + patch_size, j:j + patch_size])
		# 		submission_patches.append(image1[i:i + patch_size, j:j + patch_size])

	results = {}

	if 'PSNR' in metrics:
		# results['PSNR'] = mse2psnr(np.sum(sqerror_values) / num_dims)
		results['PSNR'] = np.sum(sqerror_values) / num_dims
	if 'MSSSIM' in metrics:
		results['MSSSIM'] = np.sum(msssim_values) / num_dims
	# if 'FID' in metrics:
	# 	results['FID'] = fid(target_patches, submission_patches)

	return results


# def fid(images0, images1):
# 	with open(os.devnull, 'w') as devnull:
# 		kwargs = {
# 			'get_codes': True,
# 			'get_preds': False,
# 			'batch_size': 100,
# 			'output': devnull}
# 		model = mmd.Inception()
# 		features0 = mmd.featurize(images0, model, **kwargs)[-1]
# 		features1 = mmd.featurize(images1, model, **kwargs)[-1]
# 		# average across splits
# 		score = np.mean(
# 			mmd.fid_score(
# 				features0,
# 				features1,
# 				splits=10,
# 				split_method='bootstrap',
# 				output=devnull))
# 	return score


def mse(image0, image1):
	return np.sum(np.square(image1 - image0))


def mse2psnr(mse):
	return 20. * np.log10(255.) - 10. * np.log10(mse)


def msssim(image0, image1):
	return MultiScaleSSIM(image0[None], image1[None])
