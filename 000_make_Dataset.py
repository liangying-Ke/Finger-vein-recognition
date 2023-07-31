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
	def iter(root, trainingSamples=[], validatingSamples=[], testingSamples=[], sub2classes={}):
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
					validatingSamples.append({'path':path, 'label':sub2classes[sub]})
				elif path in forTest:
					testingSamples.append({'path':path, 'label':sub2classes[sub]})
				else:
					trainingSamples.append({'path':path, 'label':sub2classes[sub]})
		return trainingSamples, validatingSamples, testingSamples, sub2classes
	
	trainingSamples, validatingSamples, testingSamples, sub2classes = iter(os.path.join(args.FVUSM_root, '1st_session', 'extractedvein'))
	trainingSamples, validatingSamples, testingSamples, sub2classes = iter(os.path.join(args.FVUSM_root, '2nd_session', 'extractedvein'), trainingSamples, validatingSamples, testingSamples, sub2classes)
	pickle.dump({
		'Training_Set':trainingSamples, 
		'Validating_Set':validatingSamples, 
		'Testing_Set':testingSamples,
	}, open(args.FVUSM_annotation_file, 'wb'))


def create_PLUSVein_annotation(args):
	def iter(root, Where):
		sub2classes = {}
		trainingSamples, validatingSamples, testingSamples = [], [], []
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
						validatingSamples.append({'path':path, 'label':sub2classes[identity]})
					elif path in forTest:
						testingSamples.append({'path':path, 'label':sub2classes[identity]})
					else:
						trainingSamples.append({'path':path, 'label':sub2classes[identity]})
		return trainingSamples, validatingSamples, testingSamples
	
	LED_trainingSamples, LED_validatingSamples, LED_testingSamples = iter(
		args.PLUSVein_root, os.path.join('PLUS-FV3-LED', 'PALMAR', '01'))
	LASER_trainingSamples, LASER_validatingSamples, LASER_testingSamples = iter(
		args.PLUSVein_root, os.path.join('PLUS-FV3-Laser', 'PALMAR', '01'))
	pickle.dump({
		'LED':{
			'Training_Set':LED_trainingSamples, 
			'Validating_Set':LED_validatingSamples, 
			'Testing_Set':LED_testingSamples,
		},
		'LASER':{
			'Training_Set':LASER_trainingSamples, 
			'Validating_Set':LASER_validatingSamples, 
			'Testing_Set':LASER_testingSamples,
		}
	}, open(args.PLUSVein_annotation_file, 'wb'))


if __name__== '__main__':
	args = get_argument()
	create_FVUSM_annotation(args=args)
	create_PLUSVein_annotation(args=args)
