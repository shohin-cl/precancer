#from importlib.metadata import SelectableGroups
import streamlit as st
import numpy as np
import pandas as pd
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras import backend as K
import os
import time
import io
from PIL import Image
import plotly.express as px

MODELSPATH = 'C:/Users/shohi/OneDrive/Документы/python/biology/Skin-cancer-Analyzer-master/models/'
DATAPATH = 'C:/Users/shohi/OneDrive/Документы/python/biology/Skin-cancer-Analyzer-master/data/'


def render_header():
    st.write("""
        <p align="center"> 
            <H1> Диагностика рака кожи, Precancer 
        </p>

    """, unsafe_allow_html=True)


@st.cache
def load_mekd():
    img = Image.open(DATAPATH + '/ISIC_0024312.jpg')
    return img


@st.cache
def data_gen(x):
    img = np.asarray(Image.open(x).resize((100, 75)))
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate


@st.cache
def data_gen_(img):
    img = img.reshape(100, 75)
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate


def load_models():

    model = load_model(MODELSPATH + 'model.h5')
    return model


@st.cache
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    ynew = model.predict(x_test)
    K.clear_session()
    ynew = np.round(ynew, 2)
    ynew = ynew*100
    y_new = ynew[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    K.clear_session()
    return y_new, Y_pred_classes


@st.cache
def display_prediction(y_new):

    result = pd.DataFrame({'Вероятность': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Виды болезни', 'Вероятность']
    lesion_type_dict = {2: 'Доброкачественное поражение', 4: 'Меланоцитарный невус', 3: 'Дерматофиброма',
                        5: 'Меланома', 6: 'Поражения сосудов', 1: 'Базальноклеточная карцинома', 0: 'Актинический кератоз'}
    result["Виды болезни"] = result["Виды болезни"].map(lesion_type_dict)
    return result


def main():
    st.sidebar.header('PreCancer Диагностика рака кожи')
    st.sidebar.subheader('Что вы хотите сделать:')
    page = st.sidebar.selectbox("", ["Посмотреть на пример", "Проверить себя"])

    if page == "Посмотреть на пример":
        st.header("Пример работы сайта")

        mov_base = ['Пример данных 1']
        movies_chosen = st.multiselect('Выберите данные для обратотки', mov_base)

        if len(movies_chosen) > 1:
            st.error('Данные не выбраны')
        if len(movies_chosen) == 1:
            st.success("Вы выбрали данные для обработки")
        else:
            st.info('Данные не выбраны')

        if len(movies_chosen) == 1:
            if st.checkbox('Показать пример данных '):
                st.info("Загрузка ---->>>")
                image = load_mekd()
                st.image(image, caption='Sample Data', use_column_width=True)
                st.subheader("Загрузка алгоритма с сервера")
                if st.checkbox('Загрузка Keras aлгоритма'):
                    model = load_models()
                    st.success("Алгоритм загружен")
                    if st.checkbox('Показать вероятность заболевания на примерных данныъ'):
                        x_test = data_gen(DATAPATH + '/ISIC_0024312.jpg')
                        y_new, Y_pred_classes = predict(x_test, model)
                        result = display_prediction(y_new)
                        st.write(result)
                        
    if page == "Проверить себя":

        st.header("Пройдите небольшое анкитирование")

        option = st.selectbox(
            'На какой части появилось новообразование?',
            ('Лицо', 'Шея', 'Грудная клетка', 'Живот', 'Рука', 'Спина', 'Ноги'))
        
        option1 = st.selectbox(
            'При надавливании на новообразовани, чувствуете ли вы боль?',
            ('Да, очень больно', 'Особой боли нет, но чувствую дискомфорт', 'Боли не чувствую'))
        
        option2 = st.selectbox(
            'Как давно появилось это новообразование?',
            ('1-3 мес', '3-6 мес', '6-9 мес', 'примерно год','больше года', 'больше двух лет'))
        
        option3 = st.selectbox(
            'Были ли в ваше роду люди с какими либо анкологиями?',
            ('Да, были', 'Возможно, не знаю точно', 'Нет, в роду не было таких заболеваний'))
            
        st.header("Загрузите свою фотографию")

        file_path = st.file_uploader('Загрузить изображение', type=['png', 'jpg'])

        if file_path is not None:
            x_test = data_gen(file_path)
            image = Image.open(file_path)
            img_array = np.array(image)

            st.success('Успешная загрузка файлов')
        else:
            st.info('Пожалуйста, загрузите файл изображения')

        if st.checkbox('Показать загруженное изображение'):
            st.info("Загрузка ---->>>")
            st.image(img_array, caption='Uploaded Image',
                     use_column_width=True)
            st.subheader("Загрузка алгоритма с сервера")
            if st.checkbox('Загрузка Keras aлгоритма'):
                model = load_models()
                st.success("Алгоритм загружен")
                
                if st.checkbox('Показать вероятность предсказания для загруженного изображения'):
                    y_new, Y_pred_classes = predict(x_test, model)
                    result = display_prediction(y_new)
                    st.write(result)
                   

if __name__ == "__main__":
    main()