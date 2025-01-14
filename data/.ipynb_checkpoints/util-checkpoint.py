import xml.etree.ElementTree as ET

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def parse_xml_boxes(root):
    bboxes=[]
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        cls_idx = VOC_CLASSES.index(cls_name)
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax, cls_idx]) 
    return bboxes