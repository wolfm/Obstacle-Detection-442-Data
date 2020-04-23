import numpy as np
from scipy.io import loadmat
import cv2
import os
import random
import matplotlib.pyplot as plt


def data_aug():
    # make augmentation folder
    if not os.path.exists("./data_aug_shift_annotation"):
        os.makedirs("./data_aug_shift_annotation")
    if not os.path.exists("./data_aug_shift_image_data"):
        os.makedirs("./data_aug_shift_image_data")
    if not os.path.exists("./data_aug_shift_annotation/annotationsV2_rectified"):
        os.makedirs("./data_aug_shift_annotation/annotationsV2_rectified")
    for root,dir,files in os.walk('./annotation/annotationsV2_rectified'):
        data_aug_annot_dir_prefix = "./data_aug_shift_annotation/annotationsV2_rectified\\"
        if len(root)>64:
            annot_dir_path = root
            dir_name_parsed = annot_dir_path.split("\\")[1]
            augmented_annotaton_dir = data_aug_annot_dir_prefix + dir_name_parsed + "\\ground_truth"

            for file_name in files:
                file_name_parsed = file_name.split('.')[0][:-1]
                original_image_path = './image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames'+ '/'+file_name_parsed + 'L.jpg'
                augmented_image_path = './data_aug_shift_image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames'+ '/'+file_name_parsed + 'L.jpg'

                if not os.path.exists('./data_aug_shift_image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames'):
                    os.makedirs('./data_aug_shift_image_data/video_data' + '/'+ str(dir_name_parsed) + '/'+'frames')

                augmented_annotation_path = augmented_annotaton_dir + '/' + str(file_name.split('.')[0][:])
                if not os.path.exists(augmented_annotaton_dir):
                    os.makedirs(augmented_annotaton_dir)

                img = cv2.imread(original_image_path)

                original_annotation_path = annot_dir_path + '/' + str(file_name)
                original_annotation = loadmat(original_annotation_path)['annotations']
                obstacle = original_annotation['obstacles'][0, 0]

                # only make changes to the images that have at least 1 obstacle
                if obstacle.shape[0] > 0:
                    # print('file_path:',original_image_path)
                    img,obstacle = translation(img,obstacle)
                    np.save(augmented_annotation_path, obstacle)
                    write_status = cv2.imwrite(augmented_image_path, img)

                # check if writing image is successful
                # print(write_status)
        # print(root)
    return None


def translation(img,obstacle):
    # print('original obs:',obstacle)
    row_shift = random.randint(-100,100)
    col_shift = random.randint(-100,100)
    # print('row_shift:',row_shift)
    # print('col_shift:',col_shift)
    # check boundary condition
    # if any bbox is shifted outside the frame, not considering that image-annotation pair
    for i in range(obstacle.shape[0]):
        col_cor = obstacle[i][0]
        row_cor = obstacle[i][1]
        w = obstacle[i][2]
        h = obstacle[i][3]

        if 0 < col_cor + col_shift < img.shape[1] and 0 < row_cor + row_shift <img.shape[0]:
            if col_cor + col_shift + w < img.shape[1] and row_cor + row_shift + h < img.shape[0]:
                pass
            else:
                # print('none')
                return img,obstacle
        else :
            # print('none')
            return img,obstacle

    img = np.roll(img, (row_shift, col_shift), axis=(0, 1))
    for i in range(obstacle.shape[0]):
        col_cor = obstacle[i][0]
        row_cor = obstacle[i][1]
        w = obstacle[i][2]
        h = obstacle[i][3]
        bbox_entry = np.array([[col_cor + col_shift, row_cor + row_shift, w, h]])
        # print('bbox_entry:',bbox_entry)
        if i == 0:
            bbox = bbox_entry
        else:
            np.vstack((bbox, bbox_entry))

    # print('post_obs:',bbox)
    # print('----------------------------------------------')
    return img,bbox

def load_shift_data():
    # dataset_list contains the entire dataset. It is a list of dict, and each dict correspond to each individual training image
    dataset_list = []

    # TODO: change annotation in the next line to the local directory of your annotation data set
    for root, dir, files in os.walk('./data_aug_shift_annotation/annotationsV2_rectified'):
        if len(root) > 64:

            # record is a dict contains info of 1 training image
            record = {}
            annot_dir_path = root
            dir_name_parsed = annot_dir_path.split("\\")[1]
            for file_name in files:
                file_name_parsed = file_name.split('.')[0][:-1]

                # data file path
                # TODO: change image_data to the local directory of your image dataset
                full_image_path = './image_data/video_data' + '/' + str(
                    dir_name_parsed) + '/' + 'frames' + '/' + file_name_parsed + 'L.jpg'
                record["file_name"] = full_image_path

                # data file name
                image_file_name = file_name_parsed + 'L.jpg'
                record["image_id"] = image_file_name

                # process annotation
                full_path = annot_dir_path + '/' + str(file_name)
                obstacle = np.load(full_path,allow_pickle=True)
                print(full_path)
                print(obstacle.shape)

                # annotation_list contains dict of instance in a training image
                annotation_list = []
                if obstacle.shape[0] > 0:
                    for i in range(obstacle.shape[0]):
                        bbox = obstacle[i, :]
                        bbox = bbox.tolist()
                        bbox_mode = "BoxMode.XYWH_ABS"
                        # annotation_instance is a dict that contains info of one instance in one training image
                        annotation_instance = {"bbox": bbox, "bbox_mode": bbox_mode,
                                               "category_id": 0}
                        annotation_list.append(annotation_instance)

                record["annotations"] = annotation_list

                # adding entry to the dataset dictionary
                dataset_list.append(record)
                print(len(dataset_list))
    print(len(dataset_list))
    return dataset_list

def main():
    # obs = np.load('./annotation/annotationsV2_rectified\kope67-00-00004500-00005050\ground_truth/00004605L.npy',allow_pickle=True)
    # obs = loadmat('./annotation/annotationsV2_rectified\kope67-00-00004500-00005050\ground_truth/00004725L.mat')['annotations']['obstacles'][0,0]
    # img = cv2.imread('./image_data/video_data\kope67-00-00004500-00005050/frames/00004725L.jpg')
    # print(obs)
    # for row in obs:
    #     img1 = cv2.rectangle(img, (row[0].astype(np.int), row[1].astype(np.int)),
    #                     (row[2].astype(np.int) + row[0].astype(np.int), row[3].astype(np.int) + row[1].astype(np.int)),
    #                     (0, 0, 0), 2)


    # print(obs.shape)
    # plt.imshow(img1)
    # plt.show()

    data_aug()

if __name__ == "__main__":
    main()