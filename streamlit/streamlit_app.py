import streamlit as st
import requests as requests
from PIL import Image
st.image(
            '/Users/digitalswitzerland/code/shredinc/moda/Moda-Personal-Assistant/Streamlit/home.png',
            width=700, # Manually Adjust the width of the image as per requirement
        )
st.markdown("""# Fashion AI - Find your outfit""")

form = st.form(key='my-form')
uploaded_file = form.file_uploader("Upload an image ")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

request_url = form.text_input('Or', 'Paste URL')

form.form_submit_button('Search')

import streamlit as st
import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

price_filter = st.slider('price in $', 1, 200, 75)  # min: 0$, max: 200$, default: 75


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
