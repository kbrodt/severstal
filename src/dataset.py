import albumentations as A
import torch
import torchvision

from src.utils import retrieve_img_mask


TO_TENSOR = torchvision.transforms.ToTensor()

p = 0.5
albu = A.Compose([
    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),
])


def train_transform(img, mask):
    data = albu(image=img, mask=mask)
    img, mask = data['image'], data['mask']

    return TO_TENSOR(img), TO_TENSOR(mask)


def dev_transform(img, mask):
    return TO_TENSOR(img), TO_TENSOR(mask)

    
class SeverStalDS(torch.utils.data.Dataset):
    def __init__(self, items, root, transform, preds=False):
        self.items = items
        self.root = root
        self.transform = transform
        self.preds = preds

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img, mask = retrieve_img_mask(item, self.root, preds=self.preds)
        
        return self.transform(img, mask)
    

def collate_fn(x):
    x, y = list(zip(*x))

    return torch.stack(x), torch.stack(y).long()
