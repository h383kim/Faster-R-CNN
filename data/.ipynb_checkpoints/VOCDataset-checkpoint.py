import os
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from data.util import parse_xml_boxes
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, img_set='trainval', transform=None):
        self.root = root
        self.transform = transform
        self.img_set = img_set
        
        self.annotation_path = os.path.join(self.root, f'PASCAL_VOC_{self.img_set}', 'VOCdevkit', 'VOC2007', 'Annotations')
        self.img_path = os.path.join(self.root, f'PASCAL_VOC_{self.img_set}', 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.annotations = [os.path.join(self.annotation_path, xml) for xml in sorted(os.listdir(self.annotation_path)) if not xml.startswith('.')]
        self.images = [os.path.join(self.img_path, xml) for xml in sorted(os.listdir(self.img_path)) if not xml.startswith('.')]
        
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

        target = {
            'bboxes': bboxes,
            'labels': labels
        }
        return image, target