from flask import Flask,request

import pandas as pd
import os
import csv
from glob import glob


from PIL import Image
import io
import shutil

def image_to_byte_array(image_path):
    img = Image.open(image_path)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr = imgByteArr.getvalue()

    return imgByteArr


# 채팅 필터 모델
from Chat.Chat_Filter import filter_chatting

# 이미지 유사도 임베딩 모델
from Image_similarity.Embed import get_vector
from Image_similarity.Embed import image_to_vec

# 방 사진에서 가구 이미지 크롭
from Detect_Furniture.img_detect import crop_model


def create_app():
    app = Flask(__name__)

    embedding_df = pd.DataFrame()

    @app.route('/')
    def index():
        return '안녕하세요'

    @app.route('/filter',methods=['POST'])
    def fiLter():
        chat_text = request.json['CHAT']
        res_text = filter_chatting(chat_text)

        return res_text

    @app.route('/add_furniture', methods=['POST'])
    def Add_Furniture_Image():
        img_file = request.files['img']

        save_img_path = f'./furniture_upload_folder/{img_file.filename}'
        img_file.save(save_img_path)

        print("gggggggggggggggg")
        img_vec_df = pd.DataFrame(get_vector(save_img_path)).T
        img_vec_df['image'] = img_file.filename
        img_vec_df.rename(columns=lambda x: str(x), inplace=True)

        with open('./Image_similarity/Embedding_img.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(img_vec_df.iloc[0])
            f.close()

        os.remove(save_img_path)

        return "upload_vector_done!"

    # @app.route('/img_to_byte', methods=['POST'])
    # def Image_To_Byte():

    @app.route('/image_furniture_detect', methods=['POST'])
    def Furniture_Detecter():
        img_file = request.files['img']
        print(img_file)

        save_img_path = f'./Detect_image_folder/{img_file.filename}'
        img_file.save(save_img_path)

        crop_model(save_img_path)

        image_class_list = os.listdir('./runs/detect/exp/crops')
        print(image_class_list)

        # print('C:/Users/HP/Desktop/LastProject/last/runs/detect/exp/crops/*/*.jpg')
        #
        # crop_path = 'C:/Users/HP/Desktop/LastProject/last/runs/detect/exp/crops'
        # for i,furnitures in enumerate(glob(crop_path + '/*/*.jpg')):


        crop_img_path = glob('./runs/detect/exp/crops/*/*.jpg')

        Crop_detect_list = []
        for crop_img in crop_img_path:
            crop_img = crop_img.replace("//","/")
            crop_img_byte = image_to_byte_array(crop_img)
            Crop_detect_list.append(crop_img_byte)

        print(Crop_detect_list)


        if os.path.exists('./Detect_image_folder'):
            os.remove(save_img_path)

        if os.path.exists('./runs/detect/exp/crops'):
            shutil.rmtree('./runs')

        return str(len(Crop_detect_list))




    # 가구 누를 시 이미지 유사도 계산
    @app.route('/get_image',methods=['POST'])
    def Get_Image_Embedding():
        img_file = request.files['img']
        print(img_file)

        save_img_path = f'Embedding_image_folder/{img_file.filename}'
        img_file.save(save_img_path)
        # print('bbbbbbbbbbb')

        sim_list = image_to_vec(save_img_path)
        print(sim_list)

        if os.path.exists('./Embedding_image_folder'):
            os.remove(save_img_path)

        # ## 바이트 가져오기
        # print(img_file.read())

        return sim_list

    return app
