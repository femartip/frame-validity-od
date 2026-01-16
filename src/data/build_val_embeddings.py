import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.decomposition import PCA

from zod import ZodFrames
import zod.constants as constants


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZodValDataset(Dataset):
    def __init__(self, zod_frames, frame_ids, transform):
        self.zod_frames = zod_frames
        self.frame_ids = frame_ids
        self.transform = transform

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        zod_frame = self.zod_frames[frame_id]
        image = zod_frame.get_image()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = self.transform(image)
        return frame_id, image


def build_model(device: torch.device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-dim", type=int, default=256)
    args = parser.parse_args()

    zod_frames = ZodFrames(dataset_root="./data/zod", version="full")
    
    frame_ids = sorted(list(zod_frames.get_split(constants.VAL)))
    
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  #ResNet default params
    dataset = ZodValDataset(zod_frames, frame_ids, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(DEVICE)

    all_ids = []
    all_embeds = []
    with torch.no_grad():
        for batch_ids, images in loader:
            images = images.to(DEVICE)
            embeds = model(images).cpu().numpy()
            all_ids.extend([int(b_id) for b_id in batch_ids])
            all_embeds.append(embeds)

    embeddings = np.concatenate(all_embeds, axis=0)

    pca = PCA(n_components=args.pca_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)

    columns = [f"embed_{i}" for i in range(reduced.shape[1])]
    #rows = [*vec for vec in reduced]
    df = pd.DataFrame(reduced, columns=columns, index=all_ids)

    df.to_csv("./data/image_embeddings.csv", index_label="image_id")
    

if __name__ == "__main__":
    main()
