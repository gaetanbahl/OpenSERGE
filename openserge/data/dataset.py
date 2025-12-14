from typing import Dict, Tuple, List
import os, json, random, math, cv2, numpy as np, torch
from torch.utils.data import Dataset
import sys
import glob
import pickle

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


class CityScale(Dataset):
    """
    Dataset for Sat2Graph/CityScale data.
    Expected structure:
      data_root/
        20cities/
          region_0_sat.png
          region_0_graph_gt.pickle
          ...
        data_split.json

    The graph pickle files contain a dict mapping (y,x) node coordinates to lists of (y,x) neighbors.

    Example usage:
        >>> from openserge.data.dataset import CityScale
        >>> dataset = CityScale('Sat2Graph/data/data', split='train')
        >>> sample = dataset[0]
        >>> print(sample['image'].shape)  # [3, 512, 512]
        >>> print(sample['junction_map'].shape)  # [1, 16, 16]
        >>> print(sample['offset_map'].shape)  # [2, 16, 16]
        >>> print(sample['offset_mask'].shape)  # [1, 16, 16]

    Returns:
        Sample dict with keys:
            - 'image': [3, H, W] RGB image tensor, normalized to [0, 1]
            - 'junction_map': [1, h, w] binary junction map (1 = junction present)
            - 'offset_map': [2, h, w] offset from grid cell center to exact junction location
            - 'offset_mask': [1, h, w] mask indicating which cells have junctions
            - 'meta': dict with 'region_id', 'crop_y0', 'crop_x0'
        where h = H // stride, w = W // stride
    """
    def __init__(self, data_root, split='train', img_size=512, stride=32, aug=True):
        """
        Args:
            data_root: Path to Sat2Graph/data/data directory
            split: 'train', 'valid', or 'test'
            img_size: Size of image crops (default 512)
            stride: Downsampling stride for junction/offset maps (default 32)
            aug: Whether to apply augmentation (default True)
        """
        self.root = data_root
        self.split = split
        self.img_size = img_size
        self.stride = stride
        self.aug = aug

        # Load data split
        split_path = os.path.join(data_root, 'data_split.json')
        with open(split_path, 'r') as f:
            splits = json.load(f)
        self.region_ids = splits[split]

        # Path to 20cities directory
        self.data_dir = os.path.join(data_root, '20cities')

        # Load all graphs and images
        self.samples = []
        for region_id in self.region_ids:
            img_path = os.path.join(self.data_dir, f'region_{region_id}_sat.png')
            graph_path = os.path.join(self.data_dir, f'region_{region_id}_graph_gt.pickle')

            if os.path.exists(img_path) and os.path.exists(graph_path):
                self.samples.append({
                    'region_id': region_id,
                    'img_path': img_path,
                    'graph_path': graph_path
                })

    def __len__(self):
        return len(self.samples)

    def _load_graph(self, graph_path: str) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Load graph from pickle file."""
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        return graph

    def _rasterize_graph(self, graph: Dict, crop_y0: int, crop_x0: int,
                        crop_h: int, crop_w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rasterize graph to junction map, offset map, and offset mask.

        Args:
            graph: Dict mapping (y,x) node coords to list of (y,x) neighbor coords
            crop_y0, crop_x0: Top-left corner of crop in image coordinates
            crop_h, crop_w: Height and width of crop in pixels

        Returns:
            junction_map: [1, h', w'] binary junction map (1 = junction present)
            offset_map: [2, h', w'] offset from grid cell center to exact junction location
            offset_mask: [1, h', w'] mask indicating which cells have junctions
        """
        h = crop_h // self.stride
        w = crop_w // self.stride

        junction_map = np.zeros((1, h, w), dtype=np.float32)
        offset_map = np.zeros((2, h, w), dtype=np.float32)
        offset_mask = np.zeros((1, h, w), dtype=np.float32)

        # Filter nodes that fall within the crop
        for node_coord in graph.keys():
            node_y, node_x = node_coord

            # Check if node is within crop bounds
            if not (crop_y0 <= node_y < crop_y0 + crop_h and
                   crop_x0 <= node_x < crop_x0 + crop_w):
                continue

            # Convert to crop-relative coordinates
            rel_y = node_y - crop_y0
            rel_x = node_x - crop_x0

            # Convert to grid cell coordinates
            cell_y = int(rel_y // self.stride)
            cell_x = int(rel_x // self.stride)

            # Ensure within bounds (edge case handling)
            if not (0 <= cell_y < h and 0 <= cell_x < w):
                continue

            # Mark junction as present
            junction_map[0, cell_y, cell_x] = 1.0
            offset_mask[0, cell_y, cell_x] = 1.0

            # Calculate offset from cell center to exact node location
            # Cell center in crop coordinates
            cell_center_y = (cell_y + 0.5) * self.stride
            cell_center_x = (cell_x + 0.5) * self.stride

            # Offset in pixels
            offset_y = rel_y - cell_center_y
            offset_x = rel_x - cell_center_x

            # Normalize to [-0.5, 0.5] range by dividing by stride
            offset_map[0, cell_y, cell_x] = offset_y / self.stride
            offset_map[1, cell_y, cell_x] = offset_x / self.stride

        return junction_map, offset_map, offset_mask

    def __getitem__(self, i):
        sample_info = self.samples[i]

        # Load image
        img = cv2.imread(sample_info['img_path'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # Load graph
        graph = self._load_graph(sample_info['graph_path'])

        # Random crop for training, center crop for validation/test
        if self.split == 'train' and self.aug:
            y0 = 0 if H <= self.img_size else np.random.randint(0, H - self.img_size + 1)
            x0 = 0 if W <= self.img_size else np.random.randint(0, W - self.img_size + 1)
        else:
            # Center crop
            y0 = max(0, (H - self.img_size) // 2)
            x0 = max(0, (W - self.img_size) // 2)

        # Crop image
        crop_h = min(self.img_size, H - y0)
        crop_w = min(self.img_size, W - x0)
        crop = img[y0:y0+crop_h, x0:x0+crop_w]

        # Pad if necessary
        if crop_h < self.img_size or crop_w < self.img_size:
            padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            padded[:crop_h, :crop_w] = crop
            crop = padded

        # Rasterize graph within crop
        junction_map, offset_map, offset_mask = self._rasterize_graph(
            graph, y0, x0, crop_h, crop_w
        )

        # Apply data augmentation if enabled
        if self.split == 'train' and self.aug:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                crop = np.fliplr(crop).copy()
                junction_map = np.flip(junction_map, axis=2).copy()
                offset_map = np.flip(offset_map, axis=2).copy()
                offset_map[1] = -offset_map[1]  # Flip x-offset sign
                offset_mask = np.flip(offset_mask, axis=2).copy()

            # Random vertical flip
            if np.random.rand() > 0.5:
                crop = np.flipud(crop).copy()
                junction_map = np.flip(junction_map, axis=1).copy()
                offset_map = np.flip(offset_map, axis=1).copy()
                offset_map[0] = -offset_map[0]  # Flip y-offset sign
                offset_mask = np.flip(offset_mask, axis=1).copy()

        sample = {
            'image': torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0,
            'junction_map': torch.from_numpy(junction_map),
            'offset_map': torch.from_numpy(offset_map),
            'offset_mask': torch.from_numpy(offset_mask),
            'meta': {
                'region_id': sample_info['region_id'],
                'crop_y0': int(y0),
                'crop_x0': int(x0)
            }
        }
        return sample


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m openserge.data.dataset <data_root> [dataset_type]")
        print("  dataset_type: 'roadtracer' or 'cityscale' (default: roadtracer)")
        sys.exit(1)

    data_root = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'roadtracer'

    if dataset_type == 'cityscale':
        dataset = CityScale(data_root=data_root, split='train')
    else:
        dataset = RoadTracer(data_root=data_root, split='train')

    print(f"Dataset: {dataset_type}")
    print(f"Number of samples: {len(dataset)}")

    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Junction map shape: {sample['junction_map'].shape}")
    print(f"Offset map shape: {sample['offset_map'].shape}")
    print(f"Offset mask shape: {sample['offset_mask'].shape}")

    # Print statistics
    print(f"Number of junctions in sample: {sample['junction_map'].sum().item()}")
    print(f"Image min/max: {sample['image'].min():.3f}/{sample['image'].max():.3f}")
    print(f"Offset range: [{sample['offset_map'].min():.3f}, {sample['offset_map'].max():.3f}]")