import streamlit as st
import numpy as np
import pandas as pd

# Adding title of your app

st.title('My first app')

# adding simple text
st.write('Here is a simple text')

number =st.slider('Slide me', 0,100,10 )

# print the text of number
st.write(f'You selected: {number}')

# adding a button
if st.button('Say hello'):
    st.write('Hello :)')
else:
    st.write('Goodbye')
# add radio button with options

genre = st.radio(
    "What's your favorite movie genre",
    ('Comedy', 'Drama', 'Documentary'))
# print the text of genre
st.write(f'You selected: {genre}')

# add a selectbox for options
# option = st.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone'))
# # print the text of option
# st.write('You selected:', option)
# add a drop down list on the left sidebar
option = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))
# add your whatsapp number
st.sidebar.text_input('Enter your whatsapp number')

# Add a file uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# create a line plot
# Plotting
data = pd.DataFrame({
  'first column': list(range(1, 11)),
  'second column': np.arange(number, number + 10)
})
st.line_chart(data)
