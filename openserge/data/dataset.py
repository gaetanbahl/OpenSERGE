from typing import Dict, Tuple, List, Optional
import os, json, random, math, cv2, numpy as np, torch
from torch.utils.data import Dataset
import sys
import glob
import pickle


class RoadGraphDataset(Dataset):
    """
    Base class for road graph extraction datasets.

    Provides common functionality for loading graph data, rasterizing to junction/offset maps,
    and applying data augmentation.

    Subclasses must implement:
        - _discover_samples(): Populate self.samples list
        - _load_and_preprocess_image(sample_info): Load and preprocess image for a sample
        - _get_metadata(sample_info, ...): Get sample-specific metadata
    """

    def __init__(self, data_root: str, split: str = 'train', img_size: int = 512,
                 stride: int = 32, aug: bool = True, preload: bool = False,
                 skip_edges: bool = False, normalize_mean: Tuple[float, float, float] = None,
                 normalize_std: Tuple[float, float, float] = None,
                 source_gsd: Optional[float] = None, target_gsd: Optional[float] = None):
        """
        Args:
            data_root: Path to dataset directory
            split: Dataset split ('train', 'valid'/'validation', 'test')
            img_size: Target image size (default 512)
            stride: Downsampling stride for junction/offset maps (default 32)
            aug: Whether to apply augmentation (default True)
            preload: Whether to preload all data into memory (default False)
            skip_edges: Whether to skip edge extraction (default False)
            normalize_mean: Mean for normalization (default None = no normalization)
            normalize_std: Std for normalization (default None = no normalization)
            source_gsd: Source ground sampling distance in meters/pixel (e.g., 1.0 for CityScale)
            target_gsd: Target ground sampling distance in meters/pixel (resample to this GSD)
        """
        self.root = data_root
        self.split = split
        self.img_size = img_size
        self.stride = stride
        self.aug = aug
        self.preload = preload
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.skip_edges = skip_edges
        self.source_gsd = source_gsd
        self.target_gsd = target_gsd

        # Compute GSD scale factor once (used for coordinate transforms)
        if source_gsd is not None and target_gsd is not None and source_gsd != target_gsd:
            self.gsd_scale_factor = source_gsd / target_gsd
            print(f"GSD resampling enabled: {source_gsd}m -> {target_gsd}m (scale={self.gsd_scale_factor:.3f})")

            # Memory warning for large upsampling
            if self.gsd_scale_factor > 2.0:
                print(f"WARNING: Large upsampling factor {self.gsd_scale_factor:.1f}x may use significant memory")
        else:
            self.gsd_scale_factor = 1.0

        # To be populated by subclass
        self.samples = []

        # Discover samples (implemented by subclass)
        self._discover_samples()

        # Preload data if requested
        self.preloaded_data = None
        if self.preload:
            self._preload_all_data()

    def _discover_samples(self):
        """Discover and populate self.samples. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _discover_samples()")

    def _load_and_preprocess_image(self, sample_info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess image for a sample.
        Must be implemented by subclass.

        Returns:
            img: Preprocessed image array [H, W, 3]
            preprocessing_info: Dict with preprocessing metadata (e.g., crop coords, scale factor)
        """
        raise NotImplementedError("Subclass must implement _load_and_preprocess_image()")

    def _get_metadata(self, sample_info: Dict, preprocessing_info: Dict) -> Dict:
        """Get sample-specific metadata. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _get_metadata()")

    def __len__(self):
        return len(self.samples)

    def _load_graph(self, graph_path: str) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Load graph from pickle file."""
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        return graph

    def _preload_all_data(self):
        """Preload all raw images and graphs into memory (before preprocessing)."""
        print(f"Preloading {len(self.samples)} samples into memory...")
        self.preloaded_data = []

        for sample_info in self.samples:
            # Load raw image (no preprocessing - just cv2.imread)
            raw_img = cv2.imread(sample_info['img_path'], cv2.IMREAD_COLOR)
            if raw_img is None:
                raise ValueError(f"Failed to load image: {sample_info['img_path']}")
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            # Apply GSD resampling at preload time
            if self.gsd_scale_factor != 1.0:
                h, w = raw_img.shape[:2]
                new_h = int(h * self.gsd_scale_factor)
                new_w = int(w * self.gsd_scale_factor)
                raw_img = cv2.resize(raw_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Load graph
            graph = self._load_graph(sample_info['graph_path'])

            # Store raw data - preprocessing will be done in __getitem__
            preloaded_item = {
                'raw_image': raw_img,  # Store raw (with GSD resampling if enabled), not preprocessed
                'graph': graph
            }
            preloaded_item.update(sample_info)  # Include original sample info

            self.preloaded_data.append(preloaded_item)

        print(f"Preloading complete!")

    def _rasterize_graph(self, graph: Dict, crop_y0: int, crop_x0: int,
                        crop_h: int, crop_w: int, scale_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Rasterize graph to junction map, offset map, offset mask, and edge list.

        Args:
            graph: Dict mapping (y,x) node coords to list of (y,x) neighbor coords
            crop_y0, crop_x0: Top-left corner of crop in original image coordinates
            crop_h, crop_w: Height and width of crop in pixels
            scale_factor: Scaling factor applied to coordinates (default 1.0, no scaling)

        Returns:
            junction_map: [1, h, w] binary junction map
            offset_map: [2, h, w] offset from grid cell center to exact junction location
            offset_mask: [1, h, w] mask indicating which cells have junctions
            edges: List of ((i1, j1), (i2, j2)) edge pairs in grid coordinates
        """
        h = crop_h // self.stride
        w = crop_w // self.stride

        junction_map = np.zeros((1, h, w), dtype=np.float32)
        offset_map = np.zeros((2, h, w), dtype=np.float32)
        offset_mask = np.zeros((1, h, w), dtype=np.float32)

        node_to_cell = {}

        for node_coord in graph.keys():
            node_y, node_x = node_coord

            # Apply scaling if needed (for resized images)
            scaled_y = node_y * scale_factor
            scaled_x = node_x * scale_factor

            # Check if node is within crop bounds
            if not (crop_y0 <= scaled_y < crop_y0 + crop_h and
                   crop_x0 <= scaled_x < crop_x0 + crop_w):
                continue

            # Convert to crop-relative coordinates
            rel_y = scaled_y - crop_y0
            rel_x = scaled_x - crop_x0

            # Convert to grid cell coordinates
            cell_y = int(rel_y // self.stride)
            cell_x = int(rel_x // self.stride)

            # Ensure within bounds
            if not (0 <= cell_y < h and 0 <= cell_x < w):
                continue

            # Store mapping from original node coord to grid cell
            node_to_cell[node_coord] = (cell_y, cell_x)

            degree = len(graph[node_coord])

            # Mark junction as present
            junction_map[0, cell_y, cell_x] += degree**2
            offset_mask[0, cell_y, cell_x] = 1.0

            # Calculate offset from cell center to exact node location
            cell_center_y = (cell_y + 0.5) * self.stride
            cell_center_x = (cell_x + 0.5) * self.stride

            offset_y = rel_y - cell_center_y
            offset_x = rel_x - cell_center_x

            # Normalize to [-0.5, 0.5] range
            offset_map[0, cell_y, cell_x] += offset_y / self.stride * degree**2
            offset_map[1, cell_y, cell_x] += offset_x / self.stride * degree**2

        # Average offsets if multiple nodes fall in the same cell
        offset_map /= np.maximum(junction_map, 1e-6)
        junction_map = np.clip(junction_map, 0, 1)

        # Extract edges
        edges = []
        if not self.skip_edges:
            for src_node, neighbors in graph.items():
                if src_node not in node_to_cell:
                    continue

                src_cell = node_to_cell[src_node]

                for dst_node in neighbors:
                    if dst_node not in node_to_cell:
                        continue

                    dst_cell = node_to_cell[dst_node]
                    edges.append((src_cell, dst_cell))

        return junction_map, offset_map, offset_mask, edges

    def _apply_augmentation(self, img: np.ndarray, junction_map: np.ndarray,
                           offset_map: np.ndarray, offset_mask: np.ndarray,
                           edges: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Apply data augmentation (horizontal/vertical flips).

        Returns:
            Augmented versions of all inputs
        """
        h, w = junction_map.shape[1], junction_map.shape[2]

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            junction_map = np.flip(junction_map, axis=2).copy()
            offset_map = np.flip(offset_map, axis=2).copy()
            offset_map[1] = -offset_map[1]  # Flip x-offset sign
            offset_mask = np.flip(offset_mask, axis=2).copy()

            if not self.skip_edges:
                edges = [((i1, w-1-j1), (i2, w-1-j2)) for (i1, j1), (i2, j2) in edges]

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            junction_map = np.flip(junction_map, axis=1).copy()
            offset_map = np.flip(offset_map, axis=1).copy()
            offset_map[0] = -offset_map[0]  # Flip y-offset sign
            offset_mask = np.flip(offset_mask, axis=1).copy()

            if not self.skip_edges:
                edges = [((h-1-i1, j1), (h-1-i2, j2)) for (i1, j1), (i2, j2) in edges]

        return img, junction_map, offset_map, offset_mask, edges

    def __getitem__(self, i):
        """Get a sample from the dataset."""
        # Get sample info and load data
        if self.preloaded_data is not None:
            # Use preloaded raw data
            data = self.preloaded_data[i]
            sample_info = {k: v for k, v in data.items()
                          if k not in ['raw_image', 'graph']}
            graph = data['graph']

            # Apply preprocessing to raw image (cropping/resizing happens here)
            # We pass the raw image through a modified sample_info dict
            temp_sample_info = sample_info.copy()
            temp_sample_info['_raw_image'] = data['raw_image']  # Pass raw image
            img, preprocessing_info = self._load_and_preprocess_image(temp_sample_info)
        else:
            # Load from disk
            sample_info = self.samples[i]
            img, preprocessing_info = self._load_and_preprocess_image(sample_info)
            graph = self._load_graph(sample_info['graph_path'])

        # Rasterize graph
        junction_map, offset_map, offset_mask, edges = self._rasterize_graph(
            graph,
            preprocessing_info.get('crop_y0', 0),
            preprocessing_info.get('crop_x0', 0),
            preprocessing_info.get('crop_h', img.shape[0]),
            preprocessing_info.get('crop_w', img.shape[1]),
            preprocessing_info.get('scale_factor', 1.0)
        )

        # Apply augmentation if training
        if self.split == 'train' and self.aug:
            img, junction_map, offset_map, offset_mask, edges = self._apply_augmentation(
                img, junction_map, offset_map, offset_mask, edges
            )

        # Convert to tensors and normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [C, H, W] in [0, 1]

        # Apply normalization if specified
        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
            std = torch.tensor(self.normalize_std).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

        sample = {
            'image': img_tensor,
            'junction_map': torch.from_numpy(junction_map),
            'offset_map': torch.from_numpy(offset_map),
            'offset_mask': torch.from_numpy(offset_mask),
            'edges': edges,
            'meta': self._get_metadata(sample_info, preprocessing_info)
        }

        return sample


class CityScale(RoadGraphDataset):
    """
    Dataset for Sat2Graph/CityScale data.

    Expected structure:
      data_root/
        20cities/
          region_0_sat.png
          region_0_graph_gt.pickle
          ...
        data_split.json
    """

    def _discover_samples(self):
        """Discover CityScale samples."""
        split_path = os.path.join(self.root, 'data_split.json')
        with open(split_path, 'r') as f:
            splits = json.load(f)
        region_ids = splits[self.split]

        data_dir = os.path.join(self.root, '20cities')

        for region_id in region_ids:
            img_path = os.path.join(data_dir, f'region_{region_id}_sat.png')
            graph_path = os.path.join(data_dir, f'region_{region_id}_graph_gt.pickle')

            if os.path.exists(img_path) and os.path.exists(graph_path):
                self.samples.append({
                    'region_id': region_id,
                    'img_path': img_path,
                    'graph_path': graph_path
                })

    def _load_and_preprocess_image(self, sample_info: Dict) -> Tuple[np.ndarray, Dict]:
        """Load image and apply random/center crop."""
        # Use preloaded raw image if available, otherwise load from disk
        if "_raw_image" in sample_info:
            img = sample_info["_raw_image"].copy()  # Copy to avoid modifying cached data
        else:
            img = cv2.imread(sample_info["img_path"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # GSD RESAMPLING - MUST BE FIRST (before cropping)
        gsd_scale_factor = 1.0
        if self.gsd_scale_factor != 1.0:
            original_h, original_w = img.shape[:2]
            new_h = int(original_h * self.gsd_scale_factor)
            new_w = int(original_w * self.gsd_scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gsd_scale_factor = self.gsd_scale_factor

        H, W = img.shape[:2]

        # Random crop for training, center crop otherwise
        if self.split == 'train' and self.aug:
            y0 = 0 if H <= self.img_size else np.random.randint(0, H - self.img_size + 1)
            x0 = 0 if W <= self.img_size else np.random.randint(0, W - self.img_size + 1)
        else:
            y0 = max(0, (H - self.img_size) // 2)
            x0 = max(0, (W - self.img_size) // 2)

        crop_h = min(self.img_size, H - y0)
        crop_w = min(self.img_size, W - x0)
        crop = img[y0:y0+crop_h, x0:x0+crop_w]

        # Pad if necessary
        if crop_h < self.img_size or crop_w < self.img_size:
            padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            padded[:crop_h, :crop_w] = crop
            crop = padded

        preprocessing_info = {
            'crop_y0': y0,
            'crop_x0': x0,
            'crop_h': crop_h,
            'crop_w': crop_w,
            'scale_factor': gsd_scale_factor
        }

        return crop, preprocessing_info

    def _get_metadata(self, sample_info: Dict, preprocessing_info: Dict) -> Dict:
        """Get CityScale-specific metadata."""
        return {
            'region_id': sample_info['region_id'],
            'crop_y0': preprocessing_info['crop_y0'],
            'crop_x0': preprocessing_info['crop_x0']
        }


class GlobalScale(RoadGraphDataset):
    """
    Dataset for Global-Scale Road Dataset from HuggingFace.

    Expected structure:
      data_root/
        train/1/region_0_sat.png, region_0_refine_gt_graph.p, ...
        validation/1/...
        in-domain-test/1/...
        out-of-domain/1/...
    """

    def __init__(self, data_root: str, split: str = 'train', img_size: int = 512,
                 stride: int = 32, aug: bool = True, preload: bool = False,
                 skip_edges: bool = False, use_refined: bool = True,
                 normalize_mean: Tuple[float, float, float] = None,
                 normalize_std: Tuple[float, float, float] = None,
                 source_gsd: Optional[float] = None, target_gsd: Optional[float] = None):
        self.use_refined = use_refined
        super().__init__(data_root, split, img_size, stride, aug, preload, skip_edges,
                        normalize_mean, normalize_std, source_gsd, target_gsd)

    def _discover_samples(self):
        """Discover GlobalScale samples."""
        split_dirs = {
            'train': 'train', 'validation': 'val', 'valid': 'val',
            'test': 'in-domain-test', 'in-domain-test': 'in-domain-test',
            'out-of-domain': 'out_of_domain', 'ood': 'out_of_domain'
        }

        if self.split not in split_dirs:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of: {list(split_dirs.keys())}")

        split_dir = split_dirs[self.split]
        data_dir = os.path.join(self.root, split_dir)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        for subfolder in sorted(os.listdir(data_dir)):
            subfolder_path = os.path.join(data_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for filename in sorted(os.listdir(subfolder_path)):
                if not filename.endswith('_sat.png'):
                    continue

                region_id = filename.replace('region_', '').replace('_sat.png', '')
                img_path = os.path.join(subfolder_path, filename)

                graph_filename = (f'region_{region_id}_refine_gt_graph.p' if self.use_refined
                                 else f'region_{region_id}_graph_gt.pickle')
                graph_path = os.path.join(subfolder_path, graph_filename)

                if os.path.exists(img_path) and os.path.exists(graph_path):
                    self.samples.append({
                        'region_id': region_id,
                        'subfolder': subfolder,
                        'img_path': img_path,
                        'graph_path': graph_path
                    })

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {data_dir}")

        print(f"GlobalScale {self.split} split: Found {len(self.samples)} samples")

    def _load_and_preprocess_image(self, sample_info: Dict) -> Tuple[np.ndarray, Dict]:
        """Load image and apply random/center crop."""
        # Use preloaded raw image if available, otherwise load from disk
        if "_raw_image" in sample_info:
            img = sample_info["_raw_image"].copy()  # Copy to avoid modifying cached data
        else:
            img = cv2.imread(sample_info["img_path"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # GSD RESAMPLING - MUST BE FIRST (before cropping)
        gsd_scale_factor = 1.0
        if self.gsd_scale_factor != 1.0:
            original_h, original_w = img.shape[:2]
            new_h = int(original_h * self.gsd_scale_factor)
            new_w = int(original_w * self.gsd_scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gsd_scale_factor = self.gsd_scale_factor

        H, W = img.shape[:2]

        # Random crop for training, center crop otherwise
        if self.split == 'train' and self.aug:
            y0 = 0 if H <= self.img_size else np.random.randint(0, H - self.img_size + 1)
            x0 = 0 if W <= self.img_size else np.random.randint(0, W - self.img_size + 1)
        else:
            y0 = max(0, (H - self.img_size) // 2)
            x0 = max(0, (W - self.img_size) // 2)

        crop_h = min(self.img_size, H - y0)
        crop_w = min(self.img_size, W - x0)
        crop = img[y0:y0+crop_h, x0:x0+crop_w]

        # Pad if necessary
        if crop_h < self.img_size or crop_w < self.img_size:
            padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            padded[:crop_h, :crop_w] = crop
            crop = padded

        preprocessing_info = {
            'crop_y0': y0,
            'crop_x0': x0,
            'crop_h': crop_h,
            'crop_w': crop_w,
            'scale_factor': gsd_scale_factor
        }

        return crop, preprocessing_info

    def _get_metadata(self, sample_info: Dict, preprocessing_info: Dict) -> Dict:
        """Get GlobalScale-specific metadata."""
        return {
            'region_id': sample_info['region_id'],
            'subfolder': sample_info['subfolder'],
            'crop_y0': preprocessing_info['crop_y0'],
            'crop_x0': preprocessing_info['crop_x0']
        }


class SpaceNet(RoadGraphDataset):
    """
    Dataset for SpaceNet road graph extraction data.

    Expected structure:
      data_root/
        AOI_2_Vegas_1001__rgb.png
        AOI_2_Vegas_1001__gt_graph.p
        AOI_2_Vegas_1001__gt_graph_dense.p
        ...
        dataset.json
    """

    def __init__(self, data_root: str, split: str = 'train', img_size: int = 512,
                 stride: int = 32, aug: bool = True, preload: bool = False,
                 skip_edges: bool = False, use_dense: bool = True,
                 split_file: Optional[str] = None,
                 normalize_mean: Tuple[float, float, float] = None,
                 normalize_std: Tuple[float, float, float] = None,
                 source_gsd: Optional[float] = None, target_gsd: Optional[float] = None):
        self.use_dense = use_dense
        self.split_file = split_file or os.path.join(data_root, 'dataset.json')
        super().__init__(data_root, split, img_size, stride, aug, preload, skip_edges,
                        normalize_mean, normalize_std, source_gsd, target_gsd)

    def _discover_samples(self):
        """Discover SpaceNet samples."""
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with open(self.split_file, 'r') as f:
            splits = json.load(f)

        # Map split names
        split_key = 'validation' if self.split in ['valid', 'validation'] else self.split

        if split_key not in splits:
            raise ValueError(f"Invalid split '{self.split}'. Available: {list(splits.keys())}")

        tile_ids = splits[split_key]
        graph_suffix = '_dense.p' if self.use_dense else '.p'

        for tile_id in tile_ids:
            img_path = os.path.join(self.root, f'{tile_id}__rgb.png')
            graph_path = os.path.join(self.root, f'{tile_id}__gt_graph{graph_suffix}')

            if os.path.exists(img_path) and os.path.exists(graph_path):
                self.samples.append({
                    'tile_id': tile_id,
                    'img_path': img_path,
                    'graph_path': graph_path
                })

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in {self.root}")

        print(f"SpaceNet {self.split} split: Found {len(self.samples)} samples (dense={self.use_dense})")

    def _load_and_preprocess_image(self, sample_info: Dict) -> Tuple[np.ndarray, Dict]:
        """Load image and resize to target size."""
        # Use preloaded raw image if available, otherwise load from disk
        if "_raw_image" in sample_info:
            img = sample_info["_raw_image"].copy()  # Copy to avoid modifying cached data
        else:
            img = cv2.imread(sample_info["img_path"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_h, original_w = img.shape[:2]

        # STEP 1: Apply GSD resampling FIRST (if enabled)
        gsd_scale_factor = 1.0
        if self.gsd_scale_factor != 1.0:
            gsd_h = int(original_h * self.gsd_scale_factor)
            gsd_w = int(original_w * self.gsd_scale_factor)
            img = cv2.resize(img, (gsd_w, gsd_h), interpolation=cv2.INTER_LINEAR)
            gsd_scale_factor = self.gsd_scale_factor

        # STEP 2: Resize to target img_size (existing SpaceNet behavior)
        w = img.shape[1]
        img_resized = cv2.resize(img, (self.img_size, self.img_size),
                                interpolation=cv2.INTER_LINEAR)

        # COMBINED scale factor for graph coordinates
        # Example: GSD 0.3m -> 1.0m gives gsd_scale=0.3, then resize 390->512 gives resize_scale=1.31
        # Total scale = 0.3 * 1.31 = 0.39
        resize_scale_factor = self.img_size / w
        combined_scale_factor = gsd_scale_factor * resize_scale_factor

        preprocessing_info = {
            'crop_y0': 0,
            'crop_x0': 0,
            'crop_h': self.img_size,
            'crop_w': self.img_size,
            'scale_factor': combined_scale_factor,
            'original_size': (original_h, original_w),
            'flip_y_height': original_h  # For flipping Y coordinates
        }

        return img_resized, preprocessing_info

    def _load_graph(self, graph_path: str) -> Dict:
        """Load graph and flip Y coordinates (SpaceNet uses bottom-left origin)."""
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)

        # Get the image height for this sample to flip Y coordinates
        # We need to determine the original image height
        # Extract tile_id from graph_path to get corresponding image
        tile_id = os.path.basename(graph_path).split('__gt_graph')[0]
        img_path = os.path.join(os.path.dirname(graph_path), f'{tile_id}__rgb.png')

        # Read image to get height
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img_height = img.shape[0]

        # Flip Y coordinates: new_y = img_height - old_y
        flipped_graph = {}
        for (y, x), neighbors in graph.items():
            flipped_y = img_height - y
            flipped_neighbors = [(img_height - ny, nx) for ny, nx in neighbors]
            flipped_graph[(flipped_y, x)] = flipped_neighbors

        return flipped_graph

    def _get_metadata(self, sample_info: Dict, preprocessing_info: Dict) -> Dict:
        """Get SpaceNet-specific metadata."""
        return {
            'tile_id': sample_info['tile_id'],
            'original_size': preprocessing_info['original_size'],
            'scale_factor': preprocessing_info['scale_factor']
        }


class RoadTracer(RoadGraphDataset):
    """
    Dataset for RoadTracer road graph extraction data.

    Expected structure:
      data_root/
        data/
          imagery/
            {city}_{x}_{y}_sat.png
            ...
          train_graphs_img.json

    The train_graphs_img.json file contains a list of samples with:
      - name: Image filename (e.g., "chicago_-1_0_sat.png")
      - points: List of [x, y] node coordinates
      - edges: List of [i, j] edge index pairs
    """

    def __init__(self, data_root: str, split: str = 'train', img_size: int = 512,
                 stride: int = 32, aug: bool = True, preload: bool = False,
                 skip_edges: bool = False, val_percent: float = 0.15,
                 normalize_mean: Tuple[float, float, float] = None,
                 normalize_std: Tuple[float, float, float] = None,
                 source_gsd: Optional[float] = None, target_gsd: Optional[float] = None):
        """
        Args:
            val_percent: Percentage of training cities to use for validation (default 0.15 = 15%)
                        Only used when split='train' or 'valid'/'validation'
        """
        # Test cities from README.md (15 cities)
        self.test_cities = {
            'amsterdam', 'boston', 'chicago', 'denver', 'kansas city', 'la',
            'montreal', 'new york', 'paris', 'pittsburgh', 'saltlakecity',
            'san diego', 'tokyo', 'toronto', 'vancouver'
        }
        self.val_percent = val_percent
        super().__init__(data_root, split, img_size, stride, aug, preload, skip_edges,
                        normalize_mean, normalize_std, source_gsd, target_gsd)

    def _parse_city_from_filename(self, filename: str) -> str:
        """Extract city name from filename like 'chicago_-1_0_sat.png'."""
        # Remove .png extension
        name = filename.replace('.png', '').replace('_sat', '')
        parts = name.split('_')

        # Handle multi-word cities
        if parts[0] in ['new', 'kansas', 'san']:
            return parts[0] + ' ' + parts[1]
        else:
            return parts[0]

    def _discover_samples(self):
        """Discover RoadTracer samples from train_graphs_img.json with train/val split."""
        json_path = os.path.join(self.root, 'data', 'train_graphs_img.json')

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"RoadTracer JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            all_samples = json.load(f)

        # First, collect all training cities (non-test cities)
        all_train_cities = set()
        for sample_data in all_samples:
            filename = sample_data['name']
            city = self._parse_city_from_filename(filename)
            if city not in self.test_cities:
                all_train_cities.add(city)

        # Sort cities for deterministic split
        sorted_train_cities = sorted(list(all_train_cities))

        # Split training cities into train and validation
        num_val_cities = max(1, int(len(sorted_train_cities) * self.val_percent))
        val_cities = set(sorted_train_cities[:num_val_cities])
        train_cities = set(sorted_train_cities[num_val_cities:])

        # Filter samples by split
        for sample_data in all_samples:
            filename = sample_data['name']
            city = self._parse_city_from_filename(filename)

            # Determine which split this sample belongs to
            if self.split == 'test':
                # Test split: use predefined test cities
                if city not in self.test_cities:
                    continue
            elif self.split in ['valid', 'validation']:
                # Validation split: use val_percent of training cities
                if city not in val_cities:
                    continue
            elif self.split == 'train':
                # Train split: remaining training cities
                if city not in train_cities:
                    continue
            else:
                raise ValueError(f"Invalid split: {self.split}")

            img_path = os.path.join(self.root, 'data', 'imagery', filename)

            if os.path.exists(img_path):
                self.samples.append({
                    'filename': filename,
                    'city': city,
                    'img_path': img_path,
                    'graph_path': None,  # RoadTracer doesn't use graph files, data is in JSON
                    'points': sample_data['points'],
                    'edge_indices': sample_data.get('edges', [])
                })

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in {self.root}")

        # Print split information
        cities_in_split = set([s['city'] for s in self.samples])
        print(f"RoadTracer {self.split} split: Found {len(self.samples)} samples from {len(cities_in_split)} cities")
        if self.split in ['train', 'valid', 'validation']:
            print(f"  Cities: {sorted(cities_in_split)}")

    def _convert_to_adjacency_dict(self, points: List[List[float]],
                                   edge_indices: List[List[int]]) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
        """
        Convert points and edge indices to adjacency dict format.

        Args:
            points: List of [x, y] coordinates
            edge_indices: List of [i, j] edge pairs (node indices)

        Returns:
            Adjacency dict: {(y, x): [(y1, x1), (y2, x2), ...]}
        """
        # Build adjacency dict
        adj_dict = {}

        # Initialize all nodes
        for x, y in points:
            node_key = (y, x)  # Use (y, x) format as in other datasets
            if node_key not in adj_dict:
                adj_dict[node_key] = []

        # Add edges
        for i, j in edge_indices:
            if i >= len(points) or j >= len(points):
                continue  # Skip invalid indices

            xi, yi = points[i]
            xj, yj = points[j]

            node_i = (yi, xi)
            node_j = (yj, xj)

            # Add edge (note: edges in JSON might be directional, so add both directions)
            if node_j not in adj_dict[node_i]:
                adj_dict[node_i].append(node_j)

        return adj_dict

    def __getitem__(self, i):
        """
        Get a sample from the dataset.
        Override base class to handle RoadTracer's in-memory graph format.
        """
        # Get sample info and load data
        if self.preloaded_data is not None:
            # Use preloaded raw data
            data = self.preloaded_data[i]
            sample_info = {k: v for k, v in data.items()
                          if k not in ['raw_image', 'graph']}
            graph = data['graph']

            # Apply preprocessing to raw image
            temp_sample_info = sample_info.copy()
            temp_sample_info['_raw_image'] = data['raw_image']
            img, preprocessing_info = self._load_and_preprocess_image(temp_sample_info)
        else:
            # Load from disk
            sample_info = self.samples[i]
            img, preprocessing_info = self._load_and_preprocess_image(sample_info)

            # Load graph from embedded JSON data (not from pickle file)
            graph = self._convert_to_adjacency_dict(
                sample_info['points'],
                sample_info['edge_indices']
            )

        # Rasterize graph
        junction_map, offset_map, offset_mask, edges = self._rasterize_graph(
            graph,
            preprocessing_info.get('crop_y0', 0),
            preprocessing_info.get('crop_x0', 0),
            preprocessing_info.get('crop_h', img.shape[0]),
            preprocessing_info.get('crop_w', img.shape[1]),
            preprocessing_info.get('scale_factor', 1.0)
        )

        # Apply augmentation if training
        if self.split == 'train' and self.aug:
            img, junction_map, offset_map, offset_mask, edges = self._apply_augmentation(
                img, junction_map, offset_map, offset_mask, edges
            )

        # Convert to tensors and normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Apply normalization if specified
        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
            std = torch.tensor(self.normalize_std).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

        sample = {
            'image': img_tensor,
            'junction_map': torch.from_numpy(junction_map),
            'offset_map': torch.from_numpy(offset_map),
            'offset_mask': torch.from_numpy(offset_mask),
            'edges': edges,
            'meta': self._get_metadata(sample_info, preprocessing_info)
        }

        return sample

    def _load_and_preprocess_image(self, sample_info: Dict) -> Tuple[np.ndarray, Dict]:
        """Load image and apply random/center crop (same as CityScale)."""
        # Use preloaded raw image if available, otherwise load from disk
        if "_raw_image" in sample_info:
            img = sample_info["_raw_image"].copy()  # Copy to avoid modifying cached data
        else:
            img = cv2.imread(sample_info["img_path"], cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {sample_info['img_path']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # GSD RESAMPLING - MUST BE FIRST (before cropping)
        gsd_scale_factor = 1.0
        if self.gsd_scale_factor != 1.0:
            original_h, original_w = img.shape[:2]
            new_h = int(original_h * self.gsd_scale_factor)
            new_w = int(original_w * self.gsd_scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gsd_scale_factor = self.gsd_scale_factor

        H, W = img.shape[:2]

        # Random crop for training, center crop otherwise
        if self.split == 'train' and self.aug:
            y0 = 0 if H <= self.img_size else np.random.randint(0, H - self.img_size + 1)
            x0 = 0 if W <= self.img_size else np.random.randint(0, W - self.img_size + 1)
        else:
            y0 = max(0, (H - self.img_size) // 2)
            x0 = max(0, (W - self.img_size) // 2)

        crop_h = min(self.img_size, H - y0)
        crop_w = min(self.img_size, W - x0)
        crop = img[y0:y0+crop_h, x0:x0+crop_w]

        # Pad if necessary
        if crop_h < self.img_size or crop_w < self.img_size:
            padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            padded[:crop_h, :crop_w] = crop
            crop = padded

        preprocessing_info = {
            'crop_y0': y0,
            'crop_x0': x0,
            'crop_h': crop_h,
            'crop_w': crop_w,
            'scale_factor': gsd_scale_factor
        }

        return crop, preprocessing_info

    def _get_metadata(self, sample_info: Dict, preprocessing_info: Dict) -> Dict:
        """Get RoadTracer-specific metadata."""
        return {
            'filename': sample_info['filename'],
            'city': sample_info['city'],
            'crop_y0': preprocessing_info['crop_y0'],
            'crop_x0': preprocessing_info['crop_x0']
        }

    def _preload_all_data(self):
        """
        Preload all raw images and graphs into memory.
        Override base class to handle RoadTracer's in-memory graph format.
        """
        print(f"Preloading {len(self.samples)} samples into memory...")
        self.preloaded_data = []

        for sample_info in self.samples:
            # Load raw image
            raw_img = cv2.imread(sample_info['img_path'], cv2.IMREAD_COLOR)
            if raw_img is None:
                raise ValueError(f"Failed to load image: {sample_info['img_path']}")
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            # Convert graph (points/edges already in sample_info)
            graph = self._convert_to_adjacency_dict(
                sample_info['points'],
                sample_info['edge_indices']
            )

            # Store raw data
            preloaded_item = {
                'raw_image': raw_img,
                'graph': graph
            }
            preloaded_item.update(sample_info)

            self.preloaded_data.append(preloaded_item)

        print(f"Preloading complete!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m openserge.data.dataset <data_root> [dataset_type]")
        print("  dataset_type: 'cityscale', 'globalscale', 'spacenet', or 'roadtracer' (default: cityscale)")
        sys.exit(1)

    data_root = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'cityscale'

    if dataset_type == 'cityscale':
        dataset = CityScale(data_root=data_root, split='train')
    elif dataset_type == 'globalscale':
        dataset = GlobalScale(data_root=data_root, split='train')
    elif dataset_type == 'spacenet':
        dataset = SpaceNet(data_root=data_root, split='train')
    elif dataset_type == 'roadtracer':
        dataset = RoadTracer(data_root=data_root, split='train')
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

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

    if dataset_type == 'spacenet':
        print(f"Original size: {sample['meta']['original_size']}")
        print(f"Scale factor: {sample['meta']['scale_factor']:.3f}")
