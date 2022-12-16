import streamlit as st
import requests as requests
import pandas as pd

from scipy.spatial import KDTree
from webcolors import (CSS3_HEX_TO_NAMES, hex_to_rgb)
def convert_rgb_to_names(rgb_tuple):

    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'


st.image(
           'https://i.ibb.co/2t62RcM/header2.png',
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


#request_url = form.text_input('Or', 'Paste URL')
form.form_submit_button('Search')

# Create the table using the Streamlit table() function



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





# Divide the screen into 3 columns
column3, column4, column5 = st.columns(3)

if uploaded_file is not None:
    # Create the price filter slider
    price_filter = st.slider('price in $', 1, 200, 200)  # min: 0$, max: 200$, default: 75

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



if uploaded_file is not None:
    fashion_api_url = 'https://moda-api-service-u3dpfzvyuq-ew.a.run.app/predict'
    file = {'file': uploaded_file}
    response = requests.post(url=fashion_api_url, files=file)

    prediction = response.json()['results']
    st.write(prediction)

    if 'upper' not in (prediction):
        cat1 = prediction['category']
        col1 = prediction['color']
        fit1 = prediction['fit']
        des1 = prediction['design']
        sle1 = prediction.get('sleeves','N/A')
        nec1 = prediction.get('neckline','N/A')
        len1 = prediction.get('length', 'N/A')
        fab1 = prediction['fabric']

        #col1 = convert_rgb_to_names(int(col1))

        data = {
            'category' : cat1,
            'color' : col1,
            'fit' : fit1,
            'length' :  len1,
            'design' : des1,
            'sleeves' : sle1,
            'neckline' : nec1,
            'fabric' : fab1
        }
        df = pd.DataFrame(data, index=['results']).T
        column2.dataframe(df) # Set index to False to hide the index


    else:
        upper = prediction['upper']
        lower = prediction['lower']
        cat1 = upper['category']
        col1 = upper['colors']
        fit1 = upper['fit']
        des1 = upper['design']
        sle1 = upper.get('sleeves','-')
        nec1 = upper.get('neckline','-')
        len1 = upper.get('length', '-')
        fab1 = upper['fabric']

        cat2 = lower['category']
        col2 = lower['colors']
        fit2 = lower['fit']
        des2 = lower['design']
        sle2 = lower.get('sleeves','-')
        nec2 = lower.get('neckline','-')
        len2 = lower.get('length', '-')
        fab2 = lower['fabric']

        #col1 = convert_rgb_to_names(col1)
        #col2 = convert_rgb_to_names(col2)

        data1 = {
            'category' : (cat1, cat2),
            'color' : (col1, col2),
            'fit' : (fit1, fit2),
            'length' :  (len1, len2),
            'design' : (des1, des2),
            'sleeves' : (sle1, sle2),
            'neckline' : (nec1, nec2),
            'fabric' : (fab1, fab2)
        }



        df = pd.DataFrame(data1, index=['upper', 'lower']).T
        column2.dataframe(df) # Set index to False to hide the index



# params = dict(
#     image_url1 = image_url1,
#     image_url2 = image_url2,
#     image_url3 = image_url3,
#     image_url4 = image_url4,
#     image_url5 = image_url5,
#     image_url6 = image_url6,
#     asos_url1 = asos_url1,
#     asos_url2 = asos_url2,
#     asos_url3 = asos_url3,
#     asos_url4 = asos_url4,
#     asos_url5 = asos_url5,
#     asos_url6 = asos_url6,
#     price1 = price1,
#     price2 = price2,
#     price3 = price3,
#     price4 = price4,
#     price5 = price5,
#     price6 = price6)
