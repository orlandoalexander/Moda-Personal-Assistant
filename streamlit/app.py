import streamlit as st
import requests as requests
from PIL import Image
import uuid
import webbrowser
import sys

# Import the beta_grid function
from streamlit import beta_container

ima_url1 = "https://picsum.photos/200/300"

st.image(
            'header2.png',
            width=700, # Manually Adjust the width of the image as per requirement
        )

st.markdown("<h1 style='text-align: center; color: black;'>fashion made easy</h1>", unsafe_allow_html=True)

form = st.form(key='my-form')
uploaded_file = form.file_uploader("Upload an image ", type= ['png', 'jpeg', 'jpg'])
# Display the image
# Divide the screen into 3 columns
column1, column2= st.columns(2)

gender = form.radio(
    "What's your gender",
    ('female', 'male'))

if uploaded_file is not None:
    column1.image(uploaded_file, caption="Your uploaded image", width=300)
    column2.image(ima_url1, width=300)


#request_url = form.text_input('Or', 'Paste URL')
form.form_submit_button('Search')


st.markdown("<h2 style='text-align: center; color: black;'>curated styles just for you</h2>", unsafe_allow_html=True)

# Returns from API
image_url1 = "https://picsum.photos/200/300"
image_url2 = "https://picsum.photos/200/300"
image_url3 = "https://picsum.photos/200/300"
image_url4 = "https://picsum.photos/id/237/200/300"
image_url5 = "https://picsum.photos/id/237/200/300"
image_url6 = "https://picsum.photos/id/237/200/300"

asos_url1 = 'https://www.example.com/'
asos_url2 = 'https://www.example.com/'
asos_url3 = 'https://www.example.com/'
asos_url4 = 'https://www.example.com/'
asos_url5 = 'https://www.example.com/'
asos_url6 = 'https://www.example.com/'

price1=99.99
price2=149.99
price3=199.99
price4=10
price5=25
price6=50


# Create the price filter slider
price_filter = st.slider('price in $', 1, 200, 200)  # min: 0$, max: 200$, default: 75


# Divide the screen into 3 columns
column3, column4, column5 = st.columns(3)


if price1 <= price_filter:
    column3.image(image_url1, width=200, caption=f'Price: ${price1}')
    column3.markdown(f'''
    <a href={asos_url1}><button style="background-color:White;">Buy Now</button></a>
    ''',
    unsafe_allow_html=True)

elif price4 <= price_filter:
    column3.image(image_url4, width=200, caption=f'Price: ${price4}')
    column3.markdown(f'''
    <a href={asos_url4}><button style="background-color:White;">Buy Now</button></a>
    ''',
    unsafe_allow_html=True)

# Add the second image to the second column

if price2 <= price_filter:
    column4.image(image_url2, width=200, caption=f'Price: ${price2}')
    column4.markdown(f'''
    <a href={asos_url2}><button style="background-color:White;">Buy Now</button></a>
    ''',
    unsafe_allow_html=True)

elif price5 <= price_filter:
    column4.image(image_url5, width=200, caption=f'Price: ${price5}')
    column4.markdown(f'''
    <a href={asos_url5}><button style="background-color:White;">Buy Now</button></a>
    ''',
    unsafe_allow_html=True)

# Add the third image to the third column

if price3 <= price_filter:
    column5.image(image_url3, width=200, caption=f'Price: ${price3}')
    column5.markdown(f'''
    <a href={asos_url3}><button style="background-color:White;">Buy Now</button></a>
    ''',
    unsafe_allow_html=True)

elif price6 <= price_filter:
    column5.image(image_url6, width=200, caption=f'Price: ${price6}')
    column5.markdown(f'''
    <a href={asos_url6}><button style="background-color:White;">Buy Now</button></a>
    ''',
    unsafe_allow_html=True)




#fashion_api_url = 'https://taxifare.lewagon.ai/predict'
#response = requests.get(wagon_cap_api_url, params=params)

#prediction = response.json()

#pred = prediction['fare']

#st.header(f'Find your style: {pred}')

#st.image(
           # "https://s3-us-west-2.amazonaws.com/uw-s3-cdn/wp-content/uploads/sites/6/2017/11/04133712/waterfall.jpg",
            #width=400, # Manually Adjust the width of the image as per requirement
        #)



#image = Image.open('Users/digitalswitzerland/Desktop/header.png')

#st.image(image, caption='Sunrise by the mountains')
