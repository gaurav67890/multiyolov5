import json
import cv2
from tqdm import tqdm
from pathlib import Path
import numpy as np
damage_name='dent'
mode=['train','test','val']
#mode=['val']
annt_dir='/mmdetection/data/'+damage_name+'/annotations'
img_dir='/mmdetection/data/'+damage_name+'/images'
output_dir_yolo='data/'+damage_name+'/detdata'

for m in mode:
    Path(output_dir_yolo+"/"+m).mkdir(parents=True, exist_ok=True)
    segpathimages='data/'+damage_name+'/leftImg8bit/'+m
    segpathlabels = 'data/'+damage_name+'/gtFine/' + m+'/'+m
    yoloimages='data/'+damage_name+'/detdata/images/'+m
    yololabels='data/'+damage_name+'/detdata/labels/'+m
    Path(segpathimages).mkdir(parents=True, exist_ok=True)
    Path(segpathlabels).mkdir(parents=True, exist_ok=True)
    Path(yoloimages).mkdir(parents=True, exist_ok=True)
    Path(yololabels).mkdir(parents=True, exist_ok=True)

    with open(annt_dir+'/'+damage_name+'_'+m+'.json') as f:
        data=json.load(f)
    for i in tqdm(range(len(data['images']))):
        image_id=  data['images'][i]['id']
        fn=data['images'][i]['file_name']
        img=cv2.imread(img_dir+'/'+fn)
        #img_out_path=output_dir_yolo+'/'+m+'/'+fn
        img_out_path=yoloimages+'/'+fn
        #print(img_out_path)
        cv2.imwrite(img_out_path,img)
        with open(output_dir_yolo+'/'+m+'.txt','a') as file_text:
            file_text.write(img_out_path+'\n')
        fn_text=fn[:fn.rfind('.')]+'.txt'
        height=data['images'][i]['height']
        width=data['images'][i]['width']

        mask = np.zeros((height, width), dtype='uint8')

        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==image_id:
                bbox=data['annotations'][j]['bbox']

                x_c= str((bbox[0]+0.5*bbox[2])/width)
                y_c = str((bbox[1] + 0.5 * bbox[3]) / height)
                x_w = str(bbox[2] / width)
                y_h = str(bbox[3] / height)


                with open(yololabels+'/'+fn_text, 'a') as the_file:
                    the_file.write('0 '+x_c+' '+y_c+' '+x_w+' '+y_h+'\n')

                p1 = data['annotations'][j]['segmentation'][0]

                p1 = [int(i) for i in p1]
                p2 = []
                for p in range(int(len(p1) / 2)):
                    p2.append([p1[2 * p], p1[2 * p + 1]])
                fill_pts = np.array([p2], np.int32)
                cv2.fillPoly(mask, fill_pts, 255)
        mask=mask/255
        mask=cv2.resize(mask,(1024,1024))
        img=cv2.resize(img,(1024,1024))
        _,thresh=cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        fn_new=fn[:fn.rfind('.')]+'.png'
        #print(fn_new)
        cv2.imwrite(segpathlabels + '/' + fn_new, mask)
        cv2.imwrite(segpathimages + '/' + fn, img)
