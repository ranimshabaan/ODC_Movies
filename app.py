# import helper
# import pickle
# import streamlit as st
# from tensorflow.keras.models import load_model
# import tensorflow as tf

# # interpreter = tf.lite.Interpreter('artifacts/converted_model.tflite')
# # interpreter.allocate_tensors()


# # output = interpreter.get_output_details()[0]  # Model has single output.
# # input = interpreter.get_input_details()[0]  # Model has single input.


# #model = load_model('artifacts/Model.Keras')
# model = tf.keras.models.load_model("artifacts/Model.keras")
# # Title
# st.title('Sentiment Analysis :blue[ _Movie Reviews_ ]')
# text = st.text_input('Enter your review here')




# text = helper.normalize_text(text)
# if st.button('Predict'):
#   prediction = model.predict(text)
#   if prediction:
#     st.write('Good Review')
#   else:
#     st.write('Bad Review')


import helper
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the trained model and vectorizer
model = tf.keras.models.load_model("artifacts/Model.keras")
vectorizer = pickle.load(open("artifacts/tf.pkl", 'rb'))

# Title
st.title('Sentiment Analysis :blue[_Movie Reviews_]')

# Input text box
text = st.text_input('Enter your review here')

# Normalize and predict when button is clicked
if st.button('Predict'):
    try:
        # Normalize and preprocess the text
        normalized_text = helper.normalize_text(text)
        vectorized_text = vectorizer.transform([normalized_text])  # Vectorizer expects a list of texts
        
        # Model prediction
        prediction = model.predict(vectorized_text.toarray())  # Ensure the input is in array format
        
        # Handle prediction result
        if prediction[0][0] > 0.5:  # Assuming binary classification (e.g., sigmoid activation)
            st.write('Good Review')
        else:
            st.write('Bad Review')
    except Exception as e:
        st.error(f"Error: {e}")
