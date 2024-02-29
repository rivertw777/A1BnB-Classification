import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import boto3
from PIL import Image
import os

def lambda_handler(event, context):
    image_names = [path.split("photos/")[1] for path in event]

    s3 = boto3.client('s3')    
    bucket = 'a1bnb-project'
    
    # s3 사진 다운로드
    for name in image_names:
        key = 'photos/' + name
        path = '/tmp/' + name
        s3.download_file(bucket, key, path)
        
    # 사진 가져오기
    directory = '/tmp/'
    infer_images = [Image.open(os.path.join(directory, image)) for image in image_names]

    # s3 가중치 파일 다운로드
    key = 'model/classification.h5'
    path = '/tmp/classification.h5'
    s3.download_file(bucket, key, path)
    
    # 모델 불러오기
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(8, activation='softmax')  # 클래스 수를 8개로 변경
    ])

    # 모델 추론
    model.load_weights('/tmp/classification.h5')
    
    # 예측 결과를 저장할 리스트 생성
    result = []

    # 이미지 불러오기 및 예측
    for infer_image in infer_images:
        # 이미지 크기 조정
        infer_image = infer_image.resize((224, 224))
        
        x = img_to_array(infer_image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 예측
        predictions = model.predict(x)
    
        # 예측 결과를 리스트에 추가
        result.append(predictions)
    
    return custom_result(event, result)
    
def custom_result(event, result):
    classes = {0: 'balcony', 1: 'bathroom', 2: 'bedroom', 3: 'dining_room', 4: 'exterior', 5: 'kitchen', 6: 'living_room', 7: 'swimming_pool'}
    final = {}
    for i in range(len(event)):
        # 최대 확률의 인덱스 찾기
        max_index = np.argmax(result[i][0])
        # 최대 확률 값과 클래스명을 딕셔너리에 추가
        final[event[i]] = {classes[max_index]: float(result[i][0][max_index])}
        
    return final
