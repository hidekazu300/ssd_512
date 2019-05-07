import cv2 as cv
import numpy as np
import random
import xml.etree.ElementTree as ET

NUMBER = 7   #画像1枚当たりの牌の枚数

size_x = 512
size_y = 512

mode = 10  #ロバストする確率


dict={"0":"1m","1":"2m","2":"3m","3":"4m","4":"5m","5":"6m","6":"7m","7":"8m","8":"9m","9":"1p","10":"2p","11":"3p","12":"4p","13":"5p","14":"6p","15":"7p","16":"8p","17":"9p","18":"1s","19":"2s","20":"3s","21":"4s","22":"5s","23":"6s","24":"7s","25":"8s","26":"9s","27":"east","28":"south","29":"west","30":"north","31":"white","32":"hatsu","33":"tyun"}

#trainval、valの分類
def Make_txt( save_file , datasize , percent = 0.2 ):
    path_trainval = save_file + '/trainval.txt'
    with open( path_trainval , mode='w' ) as f:
        for i in range(datasize):
            f.write(str(i)+'\n')
    
    path_val = save_file + '/val.txt'
    with open( path_val , mode='w' ) as f:
        for i in range( int(datasize*percent) ):
            f.write(str(i)+'\n')


#画像、XMLの生成
def Make_PicXML(sample_filename , save_pic_filename , save_xml_filename , robust , datasize , start=0):
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
        img=np.zeros((size_y,size_x,3))
        img=cv.rectangle(img,(0,0),(size_x,size_y),(0,128,0),cv.FILLED)
        
        for num in range(NUMBER):
            pai = cv.resize(pais[No[num]],(x,y))
            
            #牌反転処理
            if random.randint(0,1)== 0:
                pai =cv.flip(pai ,0)
            
            img[place_y:place_y+y,place_x[num]:place_x[num]+x]=pai
        
        #ロバストの実行
        if robust==1:
            img = Robust(img)
            
        #保存
        filename = save_pic_filename + '/' + str(start + i) + '.jpg'
        cv.imwrite(filename,img)
    
    
        #XMLファイルの生成
        Annotation = ET.Element('annotation')
        Folder = ET.SubElement(Annotation,'folder')
        Folder.text = 'DATASET'
        Filename = ET.SubElement(Annotation,'filename')
        Filename.text = str(i)+'.jpg'

        size = ET.SubElement(Annotation,'size')
        width = ET.SubElement(size,'width')
        width.text = str(size_x)
        height = ET.SubElement(size,'height')
        height.text = str(size_y)

        for num in range(NUMBER):
            Object = ET.SubElement(Annotation, 'object')
            name = ET.SubElement(Object, 'name')
            name.text =dict[str(No[num])] 
            bndbox = ET.SubElement(Object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(place_x[num])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(place_y)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(place_x[num] + x)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(place_y + y)
            pose = ET.SubElement(Object,'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(Object,'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(Object,'difficult')
            difficult.text = '0'
    
        tree = ET.ElementTree(element=Annotation)
        
         #保存
        filename = save_xml_filename + '/' + str( start + i) + '.xml'
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        
        if i % 500 == 0 :
            print("now complete No."+str(i))


# In[6]:


#画像のロバスト
saturation_var=0.5
brightness_var=0.5
contrast_var=0.5
lighting_std=0.5  
    
def grayscale(rgb):
    return rgb.dot([0.299, 0.587, 0.114])

def saturation(rgb):
    gs = grayscale(rgb)
    alpha = 2 * np.random.random() * saturation_var 
    alpha += 1 - saturation_var
    rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
    return np.clip(rgb, 0, 255)

def brightness(rgb):
    alpha = 2 * np.random.random() * brightness_var 
    alpha += 1 - saturation_var
    rgb = rgb * alpha
    return np.clip(rgb, 0, 255)

def contrast(rgb):
    gs = grayscale(rgb).mean() * np.ones_like(rgb)
    alpha = 2 * np.random.random() * contrast_var 
    alpha += 1 - contrast_var
    rgb = rgb * alpha + (1 - alpha) * gs
    return np.clip(rgb, 0, 255)

def lighting(img):
    cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    noise = np.random.randn(3) * lighting_std
    noise = eigvec.dot(eigval * noise) * 255
    img = np.add(img, noise)
    return np.clip(img, 0, 255)
    
def Robust(img):
    a =random.randint(0 , mode)
    if a == 0:
        img = grayscale(img)
    elif a == 1:
        img = saturation(img)
    elif a == 2:
        img = brightness(img)
    elif a == 3:
        img = contrast(img)
    elif a == 4:
        img = lighting(img)
        
    return img