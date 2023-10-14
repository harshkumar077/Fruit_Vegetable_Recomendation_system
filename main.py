import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model=tf.keras.models.load_model('trained_model.h5')
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    return np.argmax(prediction)

st.sidebar.title("Dashboard")

app_mode=st.sidebar.selectbox('Select Page',['Home',"About Project","Prediction"])

# main page 

if(app_mode=='Home'):
    st.header("FRUITS AND VEGETABLE RECOMENDATION SYSTEM")
    image_path='m.jpg'
    st.image(image_path,width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

elif(app_mode=='About Project'):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This Dataset contains Image of the folowing food items")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant")
    st.subheader("Content")
    st.text("train (100 images each)")
    st.text("test (10 images each)")
    st.text("validation (10 images each)")
    
    
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image=st.file_uploader("Choose an Image")
    if (st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index=model_prediction(test_image)
        with open("labels.txt")as f:
                content=f.readlines()
                label=[]
                for i  in content:
                
                    label.append(i[:-1])
                # st.write(content)

                st.success(f"Model is Predicting it's a {label[result_index]}")
