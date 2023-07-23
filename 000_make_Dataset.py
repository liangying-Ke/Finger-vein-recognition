import os
import glob
import random
import pickle
import argparse
from tqdm import tqdm

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--FVUSM_split', type=int, default=6)
    parser.add_argument('--PLUSVein_split', type=int, default=5)
    parser.add_argument('--FVUSM_root', type=str, default=r'C:\Datasets\FV\FV-USM')
    parser.add_argument('--FVUSM_annotation_file', type=str, default=r'.\datasets\annotations_fvusm.pkl')
    parser.add_argument('--PLUSVein_root', type=str, default=r'C:\Datasets\FV\PLUSVein-FV3\PLUSVein-FV3-ROI_combined\ROI')
    parser.add_argument('--PLUSVein_annotation_file', type=str, default=r'.\datasets\annotations_plusvein.pkl')
    return parser.parse_args([])


def create_FVUSM_annotation(args):
	def iter(root, Training_Samples=[], Validating_Samples=[], Testing_Samples=[], sub2classes={}):
		for sub in tqdm(os.listdir(root)):
			if not os.path.isdir(os.path.join(root, sub)):
				continue
			paths = glob.glob(os.path.join(root, sub, '*.jpg'))
			random.shuffle(paths)
			breakpoints = len(paths) // args.FVUSM_split
			forTest = paths[:breakpoints]
			forValidate = paths[breakpoints:breakpoints*2]
			if sub not in sub2classes:
				sub2classes[sub] = len(sub2classes)
		
			for path in paths:
				if path in forValidate:
					Validating_Samples.append({'path':path, 'label':sub2classes[sub]})
				elif path in forTest:
					Testing_Samples.append({'path':path, 'label':sub2classes[sub]})
				else:
					Training_Samples.append({'path':path, 'label':sub2classes[sub]})
		return Training_Samples, Validating_Samples, Testing_Samples, sub2classes
	
	Training_Samples, Validating_Samples, Testing_Samples, sub2classes = iter(os.path.join(args.FVUSM_root, '1st_session', 'extractedvein'))
	Training_Samples, Validating_Samples, Testing_Samples, sub2classes = iter(os.path.join(args.FVUSM_root, '2nd_session', 'extractedvein'), Training_Samples, Validating_Samples, Testing_Samples, sub2classes)
	pickle.dump({
		'Training_Set':Training_Samples, 
		'Validating_Set':Validating_Samples, 
		'Testing_Set':Testing_Samples,
	}, open(args.FVUSM_annotation_file, 'wb'))


def create_PLUSVein_annotation(args):
	def iter(root, Where):
		sub2classes = {}
		Training_Samples, Validating_Samples, Testing_Samples = [], [], []
		data_path = os.path.join(root, Where)

		for folder in tqdm(os.listdir(data_path)):
			if not os.path.isdir(os.path.join(data_path, folder)):
				continue
			paths = glob.glob(os.path.join(data_path, folder, '*.png'))
			for idx in ['02', '03', '04', '07', '08', '09']:
				identity = f'{folder}_{idx}'
				filter_paths = [path for path in paths if identity in path]
				random.shuffle(filter_paths)
				breakpoints = len(filter_paths) // args.PLUSVein_split
				forTest = filter_paths[:breakpoints]
				forValidate = filter_paths[breakpoints:breakpoints*2]

				if not identity in sub2classes:
					sub2classes[identity] = len(sub2classes)
			
				for path in filter_paths:
					if path in forValidate:
						Validating_Samples.append({'path':path, 'label':sub2classes[identity]})
					elif path in forTest:
						Testing_Samples.append({'path':path, 'label':sub2classes[identity]})
					else:
						Training_Samples.append({'path':path, 'label':sub2classes[identity]})
		return Training_Samples, Validating_Samples, Testing_Samples
	
	LED_Training_Samples, LED_Validating_Samples, LED_Testing_Samples = iter(
		args.PLUSVein_root, os.path.join('PLUS-FV3-LED', 'PALMAR', '01'))
	LASER_Training_Samples, LASER_Validating_Samples, LASER_Testing_Samples = iter(
		args.PLUSVein_root, os.path.join('PLUS-FV3-Laser', 'PALMAR', '01'))
	pickle.dump({
		'LED':{
			'Training_Set':LED_Training_Samples, 
			'Validating_Set':LED_Validating_Samples, 
			'Testing_Set':LED_Testing_Samples,
		},
		'LASER':{
			'Training_Set':LASER_Training_Samples, 
			'Validating_Set':LASER_Validating_Samples, 
			'Testing_Set':LASER_Testing_Samples,
		}
	}, open(args.PLUSVein_annotation_file, 'wb'))


if __name__== '__main__':
	args = get_argument()
	create_FVUSM_annotation(args=args)
	create_PLUSVein_annotation(args=args)