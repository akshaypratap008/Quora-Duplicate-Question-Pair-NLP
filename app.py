import os
import pickle
import re
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
import streamlit as st

from utils import prediction_pipeline


# @st.cache_resource
# def load_model():
#     model_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'model.pkl')
#     with open(model_path, 'rb') as f:
#         return pickle.load(f)


def main():
    st.set_page_config(
        page_title='Quora Duplicate Question Pair',
        page_icon='🔍',
        layout='centered',
    )

    st.title('Quora Duplicate Question Pair Detector')
    st.write(
        'This demo predicts whether two Quora-style questions are duplicates using a trained XGBoost model and handcrafted text similarity features.'
    )

    with st.expander('How this project works', expanded=True):
        st.markdown(
            '''- The model compares question lengths, shared words, token overlap, substring similarity, and fuzzy matching.
- Questions are cleaned, normalized, and compared with handcrafted features.
- The model returns whether the two questions are likely duplicates.'''
        )
        st.info('Enter two questions on the right and left, then click Predict to see whether the pair is duplicate.')

    st.subheader('Try it with your own questions')
    col1, col2 = st.columns(2)
    with col1:
        question1 = st.text_area('Question 1', height=160, placeholder='Enter the first question here')
    with col2:
        question2 = st.text_area('Question 2', height=160, placeholder='Enter the second question here')

    if st.button('Predict duplicate'):
        if not question1.strip() or not question2.strip():
            st.warning('Please enter both questions before predicting.')
            return

        prediction, probability = prediction_pipeline(question1, question2)
        prediction = prediction[0]
        probability = probability[0][0]

        label = 'Duplicate' if int(prediction) == 1 else 'Not duplicate'
        st.markdown('---')
        if int(prediction) == 1:
            st.success(f'✅ Model prediction: **{label}**')
        else:
            st.error(f'❌ Model prediction: **{label}**')

        if probability is not None:
            st.write(f'Confidence score for duplicate: **{probability:.2f}**')

    st.markdown('---')
    st.write(
        'This Streamlit app is a lightweight demo for detecting duplicate Quora-style question pairs.'
    )


if __name__ == '__main__':
    main()
