from torch.utils.data import Dataset
import subprocess,os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np 
from utils import line


DATA_DIR = 'data'
class QuickDraw(Dataset):
    def __init__(self, root, transform, mode='all', sample = 20, download = False):
        print(f"Init Dataset for {mode}-{sample} mode")
        self.root = root
        self.transform = transform
        self.sample = sample
        self.mode = mode
        self._bookkeeping_path = os.path.join(self.root, DATA_DIR,'bookkeeping' ,'quickdraw-' + mode+'-'+str(sample)+ '-bookkeeping.pkl')
        self.data_path = os.path.join(self.root, DATA_DIR, 'jsondata')
        dir_list = os.listdir(self.data_path)
        if mode == "debuging":
            dir_list=dir_list[:20]
        elif mode == "train":
            dir_list=dir_list[:int(len(dir_list)*0.6)]
        elif mode == "validation":
            dir_list=dir_list[int(len(dir_list)*0.6):int(len(dir_list)*0.8)]
        elif mode == "test":
            dir_list=dir_list[int(len(dir_list)*0.8):]
        self.splits = list(map(lambda x: os.path.splitext(x)[0],dir_list))
        if not self._check_exists() and download:
            self.download()
        self.load_bookkeeping()
        self.load_data()

    def _check_exists(self):
        if not os.path.exists(self.root):
            return False
        if not os.path.exists(self.data_path):
            return False
        return True

    def download(self):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        print('Downloading Quickdraw dataset (25Gb)')
        cmd = ['gsutil', '-m', 'cp','gs://quickdraw_dataset/full/simplified/*.ndjson',self.data_path]
        subprocess.call(cmd)

    def load_bookkeeping(self):
        if not os.path.exists(self._bookkeeping_path):
            # create bookkeeping
            labels = list(range(len(self.splits)))
            indices_to_labels = {}
            labels_to_indices = {}
            offsets = []
            index_counter = 0
            for cls_idx, cls_name in enumerate(self.splits):
                if self.sample == 'all':
                    cls_path = os.path.join(self.data_path, cls_name + '.ndjson')
                    cls_data = pd.read_json(cls_path, lines=True)
                    num_samples = cls_data.shape[0]
                else:
                    num_samples = self.sample
                labels_to_indices[cls_idx] = list(range(index_counter, index_counter + num_samples))
                for i in range(num_samples):
                    indices_to_labels[index_counter + i] = cls_idx
                offsets.append(index_counter)
                index_counter += num_samples
            bookkeeping = {
                'labels_to_indices': labels_to_indices,
                'indices_to_labels': indices_to_labels,
                'labels': labels,
                'offsets': offsets,
            }
            # Save bookkeeping to disk
            with open(self._bookkeeping_path, 'wb') as f:
                pickle.dump(bookkeeping, f, protocol=-1)
        else:
            with open(self._bookkeeping_path, 'rb') as f:
                bookkeeping = pickle.load(f)
        self._bookkeeping = bookkeeping
        self.labels_to_indices = bookkeeping['labels_to_indices']
        self.indices_to_labels = bookkeeping['indices_to_labels']
        self.labels = bookkeeping['labels']
        self.offsets = bookkeeping['offsets']

    def load_data(self):
        self.data = []
        # load cache if exist
        cache_path = f'data/data_sample/{self.mode}-{self.sample}.pkl'
        if not os.path.exists(cache_path):
            bar = tqdm(enumerate(self.splits), total=len(self.splits))
            for i,cls_name in bar:
                bar.set_postfix(data_path = cls_name)
                cls_path = os.path.join(self.data_path, cls_name + '.ndjson')
                data_df = pd.read_json(cls_path, lines=True)
                if self.sample == 'all':
                    self.data.append(data_df)
                else: 
                    self.data.append(data_df.sample(self.sample))
            self.data = pd.concat(self.data, ignore_index= True)
            self.data.to_pickle(cache_path)
            print(f'save cache to {cache_path}')        
        else:
            print(f'load cache from {cache_path}')
            self.data = pd.read_pickle(cache_path)

    def __getitem__(self, i):
        label = self.indices_to_labels[i]
        ink_array = self.data.iloc[i]['drawing']
        offset = self.offsets[label]
        np_image = self.convert_strokes_to_image(ink_array)
        if self.transform:
            features = self.transform(np_image)
        return features, label

    def __len__(self):
        return len(self.indices_to_labels)
    
    def convert_strokes_to_image(self,ink_array):
        stroke_lengths = [len(stroke[0]) for stroke in ink_array]
        total_points = sum(stroke_lengths)
        np_ink = np.zeros((total_points,3),dtype=np.int16)
        current_t = 0
        for stroke in ink_array:
            for i in [0, 1]:
                np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
            current_t += len(stroke[0])
            np_ink[current_t - 1, 2] = 1 
        # lower = np.min(np_ink[:, 0:2], axis = 0)
        upper = np.max(np_ink[:, 0:2], axis = 0)
        np_image = np.zeros((upper/2+1).astype(int),dtype=np.float32)
        for idx, stroke in enumerate(ink_array):
            for i in range(stroke_lengths[idx]-1):
                # np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
                x1=int(stroke[0][i]/2)
                y1=int(stroke[1][i]/2)
                x2 = int(stroke[0][i+1]/2)
                y2 = int(stroke[1][i+1]/2)
                points_between = line(x1,y1,x2,y2)
                for p in points_between:
                    np_image[p] = 1
        return np_image