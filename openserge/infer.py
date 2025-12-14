import argparse, os, json, math, cv2, torch, numpy as np
from .models.wrapper import OpenSERGE

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--stride', type=int, default=448)
    ap.add_argument('--junction_thresh', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--save_graph', type=str, default='graph.json')
    ap.add_argument('--device', type=str, default='cuda')
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.weights, map_location=device)
    model = OpenSERGE(k=args.k)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()

    img = cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    S = args.img_size
    T = args.stride
    nodes_global = []
    edges_global = []

    with torch.no_grad():
        for y0 in range(0, max(1, H - S + 1), T):
            for x0 in range(0, max(1, W - S + 1), T):
                tile = img[y0:y0+S, x0:x0+S]
                if tile.shape[0] < S or tile.shape[1] < S:
                    pad = np.zeros((S, S, 3), dtype=tile.dtype)
                    pad[:tile.shape[0], :tile.shape[1]] = tile
                    tile = pad
                tin = torch.from_numpy(tile).permute(2,0,1).float().unsqueeze(0)/255.0
                out = model(tin.to(device), j_thr=args.junction_thresh)
                g = out['graphs'][0]
                nodes = g['nodes'].cpu().numpy()
                edges = g['edges'].cpu().numpy()
                # shift coords by tile origin
                nodes[:,0] += x0
                nodes[:,1] += y0
                base_idx = len(nodes_global)
                nodes_global.extend(nodes.tolist())
                edges_global.extend( (edges + base_idx).tolist() )

    # TODO: de-duplicate nodes near tile borders; here we just export raw concatenation
    graph = {'nodes': nodes_global, 'edges': edges_global}
    os.makedirs(os.path.dirname(args.save_graph) or '.', exist_ok=True)
    with open(args.save_graph, 'w') as f:
        json.dump(graph, f)
    print(f'Saved graph with {len(nodes_global)} nodes and {len(edges_global)} edges to {args.save_graph}')

if __name__ == '__main__':
    main()
