import argparse, os, time, torch
from torch.utils.data import DataLoader
from .data.dataset import RoadTracerLike
from .models.wrapper import OpenSERGE
from .models.losses import openserge_losses
from .utils import *
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--backbone', type=str, default='resnet50')
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--junction_thresh', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=None, help='k for kNN prior; None=complete')
    ap.add_argument('--device', type=str, default='cuda')
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ds = RoadTracerLike(args.data_root, 'train', img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = OpenSERGE(backbone=args.backbone, k=args.k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            images = batch['image'].to(device)
            targets = {
                'junction_map': batch['junction_map'].to(device),
                'offset_map': batch['offset_map'].to(device),
                'offset_mask': batch['offset_mask'].to(device),
                'edge_lists': []  # TODO: create aligned (src,dst,y) per-sample
            }
            out = model(images, j_thr=args.junction_thresh)
            losses = openserge_losses(out, targets)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix({k: float(v.detach().cpu()) for k,v in losses.items()})
        # Save checkpoint
        ckpt = {'model': model.state_dict(), 'args': vars(args), 'epoch': epoch}
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(ckpt, f'checkpoints/openserge_epoch{epoch}.pt')

if __name__ == '__main__':
    main()
