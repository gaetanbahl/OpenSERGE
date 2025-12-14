from typing import Dict, Tuple
import os, json, random, math, cv2, numpy as np, torch
from torch.utils.data import Dataset
from ..utils import build_grid
import sys
import glob

# https://github.com/mitroadmaps/roadtracer/tree/master/dataset

class RoadTracer(Dataset):
    """Dataset stub.
    Expected structure:
      data_root/
        imagery/*.png
        graphs/*.graph
        testsat/*.png
    ------------
    If data_root/graphs_crops.json exists, it contains precomputed graph crops for training.
    The graph crops contain point coordinates for each 4096x4096 image tile found in the imagery folder.
    If it does not exist, it will be created.
    ------------
    You must implement: load_graph, rasterize_cells, and sampler to produce:
      - junction_map: [1,h',w'] binary
      - offset_map: [2,h',w'] in [-0.5,0.5]
      - offset_mask: [1,h',w']
    """
    def __init__(self, data_root, split='train', img_size=512, stride=32, aug=True):
        self.root = data_root
        self.split = split
        self.img_size = img_size
        self.stride = stride
        self.aug = aug

        self.img_dir = os.path.join(data_root, 'imagery')
        self.img_names = [p for p in os.listdir(self.img_dir) if p.endswith('.png')]
        self.graphs = glob.glob(os.path.join(data_root, 'graphs', '*.graph'))

        # Check if we have precomputed crops
        crops_path = os.path.join(data_root, f'train_graphs_img.json')
        if os.path.exists(crops_path):
            with open(crops_path, 'r') as f:
                self.crops = json.load(f)

                # Convert this list of dicts to a dict indexed by image name
                crops_dict = {}
                for item in self.crops:
                    img_name = item['name']
                    crops_dict[img_name] = item
                self.crops = crops_dict
        else:
            pass
            # self.crops = self.create_graphs_crops()
            # with open(crops_path, 'w') as f:
            #     json.dump(self.crops, f)

    def create_graphs_crops(self):
        pass


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # Get graph for this image
        graph = self.crops.get(img_name, [])
        print(img_name)
        print(self.crops.keys())
        print(graph)

        # Random crop
        y0 = 0 if H<=self.img_size else np.random.randint(0, H-self.img_size+1)
        x0 = 0 if W<=self.img_size else np.random.randint(0, W-self.img_size+1)
        crop = img[y0:y0+self.img_size, x0:x0+self.img_size]

        h = self.img_size // self.stride
        w = self.img_size // self.stride
        
        junction_map = np.zeros((1, h, w), np.float32)
        offset_map = np.zeros((2, h, w), np.float32)
        offset_mask = np.zeros((1, h, w), np.float32)

        sample = {
            'image': torch.from_numpy(crop).permute(2,0,1).float()/255.0,
            'junction_map': torch.from_numpy(junction_map),
            'offset_map': torch.from_numpy(offset_map),
            'offset_mask': torch.from_numpy(offset_mask),
            #'meta': {'id': id_, 'xy0': (int(x0), int(y0))}
        }
        return sample


if __name__ == '__main__':
    dataset = RoadTracer(data_root=sys.argv[1], split='train')
    sample = dataset[0]
    print(sample['image'].shape)
    print(sample['junction_map'].shape)
    print(sample['offset_map'].shape)
    print(sample['offset_mask'].shape)