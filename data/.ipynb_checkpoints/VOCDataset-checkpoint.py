import os
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, img_set='trainval', transform=None):
        self.root = root
        self.transform = transform
        self.img_set = img_set
        
        self.annotation_path = os.path.join(self.root, f'VOC{self.img_set}_06-Nov-2007', 'VOCdevkit', 'VOC2007', 'Annotations')
        self.img_path = os.path.join(self.root, f'VOC{self.img_set}_06-Nov-2007', 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.annotations = [os.path.join(self.annotation_path, xml) for xml in sorted(os.listdir(self.annotation_path)) if not xml.startswith('.')]
        self.images = [os.path.join(self.img_path, xml) for xml in sorted(os.listdir(self.img_path)) if not xml.startswith('.')]
        self.annotations = self.annotations[3000:4000]
        self.images = self.images[3000:4000]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image\
        # Read the image using Pillow
        image = Image.open(self.images[idx])
        image = np.array(image)
        # Load annotation
        tree = ET.parse(self.annotations[idx])
        gt_bboxes_labels = parse_xml_boxes(tree.getroot())
        gt_bboxes_labels = np.array(gt_bboxes_labels)
        bboxes, labels= gt_bboxes_labels[:, :4], gt_bboxes_labels[:, -1]
        # Apply transformation if any
        if self.transform:
            image = self.transform(image)
        # Convert data to tensors
        bboxes = torch.tensor(bboxes).float()
        labels = torch.tensor(labels).long()
        
        return image, bboxes, labels