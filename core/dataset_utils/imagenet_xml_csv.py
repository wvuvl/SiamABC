import itertools
import xml.etree.ElementTree as ET
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import pandas as pd
import os

class Create_ImageNetDataset:
    
    def __init__(self, imgs_dir, ann_dir, save_path):
        '''
        loading images and annotation from imagenet
        @imgs_dir: images path
        @ann_dir: annotations path
        @isTrain: True: Training, False: validation
        @val_ratio: validation data ratio
        @dbg: For visualization
        '''

        if not Path(imgs_dir).is_dir():
            logger.error('{} is not a valid directory'.format(imgs_dir))

        self._imgs_dir = Path(imgs_dir)
        self._ann_dir = Path(ann_dir)
        self._kMaxRatio = 1
        self._list_of_annotations = self.__loadImageNetDet()
        self._data_fetched = []  # for debug purposes
        assert len(self._list_of_annotations) > 0, 'Number of valid annotations is {}'.format(len(self._list_of_annotations))

        data_len = len(self._list_of_annotations) 
        sample_arr = []    
        for i in tqdm(range(data_len),desc='Preprocessing ImageNet Dataset'): 
            sample = self.get_sample(i)
            if sample is not None:
                sample_arr.append(sample)
        
        df = pd.DataFrame(sample_arr, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
        df.to_csv(save_path)
        
    def get_sample(self, idx):
        """Get the current idx data
        @idx: Current index for the data
        """
        return self.load_bbox(idx) 

    
    def __loadImageNetDet(self):
        ''' Loads all the image annotations files '''

        subdirs = [x.parts[-1] for x in self._imgs_dir.iterdir() if x.is_dir()]

        self._subdirs = subdirs
        num_annotations = 0
        list_of_annotations_out = []

        for i, subdir in enumerate(subdirs):
            ann_files = self._ann_dir.joinpath(subdir).glob('*.xml')
            logger.info('Loading {}/{} - annotation file from folder = {}'.format(i + 1, len(subdirs), subdir))
            for ann in ann_files:
                list_of_annotations, num_ann_curr = self.__load_annotation_file(ann)
                num_annotations = num_annotations + num_ann_curr
                if len(list_of_annotations) == 0:
                    continue
                list_of_annotations_out.append(list_of_annotations)

        all_annotations = list(itertools.chain.from_iterable(list_of_annotations_out))
        # random.shuffle(all_annotations)

        logger.info('+' * 60)
        logger.info("Found {} annotations from {} images"
                    " ({:.2f} annotations/image)".format(num_annotations,
                                                         len(list_of_annotations_out),
                                                         (num_annotations / len(list_of_annotations_out))))

        return all_annotations

    def __load_annotation_file(self, annotation_file):
        """ Loads the bounding box annotations in xml file
        @annotation_file: annotation file (.xml), which contains
        bounding box information
        """

        list_of_annotations = []
        num_annotations = 0
        root = ET.parse(annotation_file).getroot()
        folder = root.find('folder').text
        filename = root.find('filename').text
        
        size = root.find('size')
        disp_width = int(size.find('width').text)
        disp_height = int(size.find('height').text)

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            width = xmax - xmin
            height = ymax - ymin

            kMaxRatio = self._kMaxRatio
            if width > (kMaxRatio * disp_width) or height > (kMaxRatio * disp_height):
                continue

            if ((xmin < 0) or (ymin < 0) or (xmax <= xmin) or (ymax <= ymin)):
                continue
            
            
            
            objAnnotation = annotation()
            objAnnotation.setbbox(xmin, xmax, ymin, ymax)
            objAnnotation.setWidthHeight(disp_width, disp_height)
            objAnnotation.setImagePath(Path(folder).joinpath(filename))
            list_of_annotations.append(objAnnotation)
            num_annotations = num_annotations + 1

        return list_of_annotations, num_annotations

    def load_bbox(self, idx):
        random_ann = self._list_of_annotations[idx]
        anno = random_ann.bbox
        img_path = str(self._imgs_dir.joinpath(random_ann.image_path.with_suffix('.JPEG')))
        
        if os.path.exists(img_path) == False: 
            logger.info('path does not exist, skipping')
            return None
        
        x = int(anno[0])
        y = int(anno[1])
        w = int(anno[2] - anno[0])
        h = int(anno[3] - anno[1])
    
        bbox_exist = 1
        bbox_border = int(x<=0 or y<=0 or (x+w)>=random_ann.shape[0]-1 or (y+h)>=random_ann.shape[1]-1)
    
        return [str(idx), str(idx), 0, img_path, [x,y,w,h], random_ann.shape, 'ImageNet2014', bbox_exist, bbox_border ] 

        
class annotation:
    
    def __init__(self):
        """Annotation class stores bounding box, image path"""
        self.bbox = [0, 0, 0, 0]
        self.image_path = []
        self.disp_width = 0
        self.disp_height = 0
        self.shape = [self.disp_width, self.disp_height]

    def setbbox(self, x1, x2, y1, y2):
        """ set the bounding box  """
        self.bbox = [x1, y1, x2, y2]

    def setImagePath(self, img_path):
        """ set the image path """
        self.image_path = img_path

    def setWidthHeight(self, disp_width, disp_height):
        """ set width and height """
        self.disp_width = disp_width
        self.disp_height = disp_height
        self.shape = [self.disp_width, self.disp_height]

    def __repr__(self):
        return str({'bbox': self.bbox, 'image_path': self.image_path,
                    'w': self.disp_width, 'h': self.disp_height})
        

if __name__ == '__main__':
    
    imagenet_path = r'/new_local_storage/zaveri/SOTA_Tracking_datasets/ILSVRC2014_DET_train'
    img_dir = Path(imagenet_path).joinpath('images')
    ann_dir = Path(imagenet_path).joinpath('gt')
    db = Create_ImageNetDataset(str(img_dir), str(ann_dir), "imagenet2014.csv")
