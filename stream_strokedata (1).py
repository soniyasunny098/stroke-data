 
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('stroke_data')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('brainomixstroke.jpg')
    image_office = Image.open('stroke-recovery-timeline.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict if a patient is likely to get a stroke or not')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Predicting Stroke Disease")
    if add_selectbox == 'Online':
        id=st.number_input('id' , min_value=1, max_value=72943, value=1)
        age =st.number_input('age',min_value=0.08, max_value=82.0, value=0.08)
        avg_glucose_level = st.number_input('avg_glucose_level', min_value=55.0, max_value=291.05, value=55.0)
        bmi = st.number_input('bmi', min_value=10.1, max_value=97.6, value=10.1)
        hypertension=st.selectbox('hypertension',[0,1])
        heart_disease=st.selectbox('heart_disease',[ 0,1])
        gender = st.selectbox('gender', ['Male', 'Female','Other'])
        ever_married = st.selectbox('ever_married', ['yes', 'no'])
        work_type = st.selectbox('work_type', ['private', 'self-employed','children','govt job','never worked'])
        Residence_type = st.selectbox('Residence_type', ['urban', 'rural'])
        smoking_status = st.selectbox('smoking_status', ['never smoked', 'formerly smoked','smokes','unknown'])
        output=""
        input_dict={'id':id,'age':age,'avg_glucose_level':avg_glucose_level,'bmi':bmi,'hypertension': hypertension,'heart_disease':heart_disease,'gender' : gender,'ever_married':ever_married,'work_type':work_type,'Residence_type':Residence_type,'smoking_status':smoking_status}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
            if output == '0':
              output="YOU WILL NOT GET A STROKE"
            else:
              output="YOU WILL GET A STROKE"
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
