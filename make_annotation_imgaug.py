
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import random
import xml.etree.ElementTree as ET
from imgaug import augmenters as iaa
import imgaug as ia


NUMBER = 7   #画像1枚当たりの牌の枚数

size_x = 512
size_y = 512

dict={"0":"1m","1":"2m","2":"3m","3":"4m","4":"5m","5":"6m","6":"7m","7":"8m","8":"9m","9":"1p","10":"2p","11":"3p","12":"4p","13":"5p","14":"6p","15":"7p","16":"8p","17":"9p","18":"1s","19":"2s","20":"3s","21":"4s","22":"5s","23":"6s","24":"7s","25":"8s","26":"9s","27":"east","28":"south","29":"west","30":"north","31":"white","32":"hatsu","33":"tyun"}


# In[2]:


aug1 = iaa.Dropout(p=0.2)
aug2 = iaa.AverageBlur(k=(5, 20))
aug3 = iaa.Add((-40, 40), per_channel=0.5)
aug4 = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
aug5 = iaa.Affine(rotate=(0,20))


# In[3]:


#画像のロバスト
def augment( img , bb , aug ):
    # 画像とバウンディングボックスを変換
    aug_img = aug.augment_image( img ) 
    aug_bb = aug.augment_bounding_boxes([bb])[0].remove_out_of_image().cut_out_of_image()
    
    '''
    # バウンディングボックスと画像を重ねる
    image_before = bb.draw_on_image(img, thickness=2, color=[255, 0,0])
    image_after = aug_bb.draw_on_image(aug_img, thickness=2, color=[0, 255, 0])

    # 変換前後の画像を描画
    fig = plt.figure()
    fig.add_subplot(121).imshow(image_before)
    fig.add_subplot(122).imshow(image_after)
    plt.show()
    '''
    return aug_img , aug_bb
    
#XMLファイルの生成
def To_Xml( filename , bb , save_dir = ""):
        Annotation = ET.Element('annotation')
        Folder = ET.SubElement(Annotation,'folder')
        Folder.text = 'DATASET'
        Filename = ET.SubElement(Annotation,'filename')
        Filename.text = filename +'.jpg'

        size = ET.SubElement(Annotation,'size')
        width = ET.SubElement(size,'width')
        width.text = str(bb.shape[0])
        height = ET.SubElement(size,'height')
        height.text = str(bb.shape[1])

        for i in range(len(bb.bounding_boxes)):
            Object = ET.SubElement(Annotation, 'object')
            name = ET.SubElement(Object, 'name')
            name.text = str(bb.bounding_boxes[i].label)
            bndbox = ET.SubElement(Object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(bb.bounding_boxes[i].x1))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(bb.bounding_boxes[i].y1))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(bb.bounding_boxes[i].x2))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(bb.bounding_boxes[i].y2))
            pose = ET.SubElement(Object,'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(Object,'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(Object,'difficult')
            difficult.text = '0'
    
        tree = ET.ElementTree(element=Annotation)
        
        #保存
        filename = save_dir  +'/'+filename + '.xml'
        tree.write(filename, encoding='utf-8', xml_declaration=True)


# In[4]:


#画像、XMLの生成
def Make_PicXML_2(sample_filename , save_pic_filename , save_xml_filename , aug_modes , datasize , start=0):
    No = []
    place_x = []
    place_y = 0
    
    #画像の読み込み
    pais=[]
    for i in range(34):
        filename = sample_filename + '/' +str(i)+'.jpg'
        pais.append(cv.imread(filename))
      
    sample_height = pais[0].shape[0]
    sample_width = pais[0].shape[1]
    
    for i in range(datasize):

        #サイズの倍率（90～100）
        magni = min(size_x/NUMBER/sample_width , size_y/sample_height)*random.randint(90,100)/100
    
        x = int(magni * sample_width)
        y = int(magni * sample_height)
        
        #牌の種類の決定
        No=[]
        for num in range(NUMBER):
            No.append(random.randint(0,33))
            
        #場所の決定
        place_x=[]
        place_x.append(random.randint( 0,int(size_x - x*NUMBER)))
        for num in range(NUMBER):
            place_x.append(place_x[0]+ x*(num+1))
    
        place_y=random.randint(1,int(size_y - y))
    
        #画像の生成
        img=np.zeros((size_y,size_x,3),dtype=np.uint8)
        img=cv.rectangle(img,(0,0),(size_x,size_y),(0,128,0),cv.FILLED)
        
        for num in range(NUMBER):
            pai = cv.resize(pais[No[num]],(x,y))
            
            #牌反転処理
            if random.randint(0,1)== 0:
                pai =cv.flip(pai ,0)
            
            img[place_y:place_y+y,place_x[num]:place_x[num]+x]=pai
       
        #bb型に変換
        boxes = []
        for num in range(NUMBER):
            boxes.append(ia.BoundingBox(x1=place_x[num], y1=place_y, x2=place_x[num]+x, y2=place_y+y,label=dict[str(No[num])]))
        bb = ia.BoundingBoxesOnImage( boxes , shape = (size_x , size_y , 3))

        #アーグメーションの実行
        for mode in range(len(aug_modes)):
            img_aug , bb_aug = augment( img , bb , aug_modes[mode])
            
            #保存
            '''境界線を記入した画像を生成
            for num in range(len(bb_aug.bounding_boxes)):
                img_aug = cv.rectangle(img_aug, (int(bb_aug.bounding_boxes[num].x1), int(bb_aug.bounding_boxes[num].y1)),(int(bb_aug.bounding_boxes[num].x2),int(bb_aug.bounding_boxes[num].y2)), (0, 0, 0) , thickness=8)
            '''
            
            filename = save_pic_filename + '/' + str(start + i) + '_' + str(mode) + '.jpg'
            cv.imwrite( filename , img_aug )

            #XMLファイルの生成、保存
            To_Xml( str(start + i) + '_' + str(mode) , bb_aug , save_dir = save_xml_filename )
        
        if i % 500 == 0:
            print("now complete No."+str(i))


# In[5]:


#Make_PicXML_2( 'sample' , 'test' , 'test' , [aug1,aug2,aug5], 500 )


# In[6]:


import os
import random
def Make_txt( filename , savefile , percent=0.2):
    flist = os.listdir(filename)
    vals =random.sample(flist, int(len(flist)*percent))
    
    #保存
    path_trainval = savefile + '/trainval.txt'
    with open( path_trainval , mode='w' ) as f:
        for name in flist:
            f.write(str(os.path.splitext(name)[0])+'\n')
            
    path_val = savefile + '/val.txt'
    with open( path_val , mode='w' ) as f:
        for val in vals:
            f.write(os.path.splitext(val)[0]+'\n')


# In[7]:


#Make_txt('DATASET/JPEGImages','DATASET')

