import cv2
import random

#original path of images that wait to be cut
imgs = ['img ({}).jpg'.format(i) for i in range(1,709)]
original_dataset_dir = "C:/Users/21300/Desktop/Dataset/Real_scene_dataset/img/MONACO/"  #what you need to change
cropped_img_num = 1

for img_name in imgs:
    #randomly select 6 subareas in the original image, cut them out later
    selected_area_x_y = [
        [random.randint(150, 284), random.randint(0,384)],
        [random.randint(540, 674), random.randint(0,384)],
        [random.randint(150, 284), random.randint(640,1024)],
        [random.randint(540, 674), random.randint(640,1024)],
        [random.randint(150, 284), random.randint(1280,1664)],
        [random.randint(540, 674), random.randint(1280,1664)]
    ]

    img_path = original_dataset_dir + img_name
    img = cv2.imread(img_path)

    # cut out 6 images mentioned above
    for selected_area in selected_area_x_y:
        y, x = selected_area
        cropped = img[y:(y+256), x:(x+256)]  # The clipping coordinates are[y0:y1, x0:x1]
        #store images
        cv2.imwrite("C:/Users/21300/Desktop/Dataset/Real_scene_dataset/cropped_img/{}.jpg".format(cropped_img_num), cropped)
        cropped_img_num += 1
        if cropped_img_num % 100 == 0:
            print('Having produced: ' + str(cropped_img_num) + ' images')