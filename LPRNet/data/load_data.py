import torch
from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os
import glob
import os
import os.path as path


class LPRDataset(Dataset):
    def __init__(self, img_dir, imgSize, PreprocFun=None):
        self.img_dir = img_dir
        self.img_datas = []
        self.img_size = imgSize
        self.anno_dict = {
            "provinces":["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"],
            "alphabets": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                    'X', 'Y', 'Z', 'O'],
            "ads":['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
            }

    
        for f_path in glob.glob(self.img_dir + "/*.jpg"):
            self.img_datas.append((f_path, path.splitext(path.basename(f_path))[0]))

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_datas)

    def __getitem__(self, index):
        file_path = self.img_datas[index][0]
        image = cv2.imread(file_path)
        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)

        # get image
        image = self.PreprocFun(image)

        # get lp_number for show
        # lp_number = self.get_lp_number(self.img_datas[index][0])

        lp_number_label = self.img_datas[index][1].split("-")[4].split("_")

        return image, lp_number_label, len(lp_number_label)

    #region get_lp_number
    # def get_lp_number(self,file_name):
    #     provinces = self.anno_dict["provinces"]
    #     alphabets = self.anno_dict["alphabets"]
    #     ads       = self.anno_dict["ads"]
        
    #     lp_number = file_name.split("-")[4].split("_")

    #     prov_value = provinces[int(lp_number[0])]
    #     alpha_value = alphabets[int(lp_number[1])]
    #     index_list = [int(item) for item in lp_number[2:]]
    #     ads_value = [ads[index] for index in index_list]
        
    #     return prov_value + alpha_value + "".join(ads_value)
    #endregion get_lp_number

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    # def check(self, label):
    #     if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
    #             and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
    #         print("Error label, Please check!")
    #         return False
    #     else:
    #         return True
        
def collate_fn(batch): # convert all ndarrays in the current batch to tensor
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.append(label)
        lengths.append(length)
    labels = np.asarray(labels).astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)
        
if __name__ == "__main__":
    train_dir = "/datav/shared/dataset/CCPD2020/ccpd_green/train"
    train_dataset = LPRDataset(train_dir, (94, 24)) # wh
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, collate_fn = collate_fn)
    print('data length is {}'.format(len(train_dataset)))
    for imgs, labels, lengths in train_dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))      
        break
    
    """ 
    unit test result:
    image batch shape is torch.Size([128, 3, 24, 94])
         ..., glance
         [ 0.3242,  0.3945,  0.4414,  ..., -0.3086, -0.4258, -0.4180]
    label batch shape is torch.Size([128, 8])
         ... glance
         ([[ 0.,  0.,  3.,  ..., 31., 25., 33.],
         [ 0.,  0.,  3.,  ..., 31., 31., 30.],

    label length is 128
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 
    
     """
