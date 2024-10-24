# -*- coding: utf-8 -*-
"""
Created on Tue Nov  18 October 2024

@author: Nihar Panchal
"""

import requests
import streamlit as st
import util
import pickle
from util import get_estimated_price
from util import load_saved_artifacts
#from util import __model

#

import pickle
import json
import numpy as np

__locality_name = None
__data_columns = None
__model = None


def get_estimated_price(locality_name, sqft, bhk):
        try:
                loc_index = __data_columns.index(locality_name.lower())
        except:
                loc_index = -1

        x = np.zeros(643)
        x[0] = sqft
        x[1] = bhk
        if loc_index >= 0:
                x[loc_index] = 1

        return round(__model.predict([x])[0], 2)
        # return round(__model.predict(([x])[0],2)


def load_saved_artifacts():
        print("loading saved artifacts...start")
        global __data_columns
        global __locality_name

        with open("column.json", "r") as f:
                __data_columns = json.load(f)['data_columns']
                __locality_name = __data_columns[2:]  # first 2 columns are sqft, bhk

        global __model
        if __model is None:
                with open('maharashtra_region_model.pickle', 'rb') as f:
                        __model = pickle.load(f)
        print("loading saved artifacts...done")


def get_location_names():
        return __locality_name


def get_data_columns():
        return __data_columns


if __name__ == '__main__':
        load_saved_artifacts()
        print(get_location_names())
        print(get_estimated_price('Moshi', 900, 3))
        print(get_estimated_price('Hadapsar', 900, 3))
        print(get_estimated_price('Talegaon', 600, 2))  # other location
        # print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
#

#############

from streamlit_lottie import st_lottie
def load_lottiefiles(filepath: str):
        with open(filepath, "r") as f:
                return json.load(f)

def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
                return None
        return r.json()

animation = load_lottiefiles("animation.json")
st_lottie(animation, key="anim", quality="low", loop=True, height=200)

############



html = """
    <div style="background:#025246 ;padding:15px">
    <h1 style="color:white;text-align:center;"> Maharashtra's House Price Prediction</h1>
    </div>
    """
st.markdown('https://github.com/mavericknihar')
st.markdown(html, unsafe_allow_html = True )


sqft=st.number_input('Enter carpet area in sqft', 375,5000)
bhk=st.select_slider('Choose number of BHK', [1,2,3,4,5])
locality_name = st.selectbox('Choose/Write the location name', ['aamrai', 'adaigaon', 'additional m.i.d.c', 'adgaon', 'adharwadi', 'adivali', 'agripada', 'airoli', 'ajni medical colony', 'akurdi', 'akurli', 'alandi', 'allyali', 'amar nagar', 'amar nagar road', 'ambarnath', 'ambegaon', 'ambegaon 1', 'ambegaon budruk', 'ambegaon pathar', 'ambernath east', 'ambernath west', 'ambivali', 'ambivli', 'amboli', 'amboli ', 'anand nagar', 'anand tirth nagar', 'anandwalli gaon', 'andheri east', 'andheri west', 'anil nagar', 'anjurdive', 'antarli', 'antop hill', 'asangaon', 'ashirwad nagar', 'ashok nagar', 'atgaon', 'aundh', 'ayodhya nagar', 'azamshah layout', 'badlapur', 'badlapur east', 'badlapur gaon', 'badlapur west', 'bajaj nagar', 'balaji nagar', 'balapur', 'balewadi', 'balkum', 'balkum road', 'bandra east', 'bandra kurla complex', 'bandra west', 'baner', 'bavdhan', 'beed bypass padegaon', 'belapur', 'beltarodi', 'besa', 'besa manish nagar road', 'besa pipla road', 'beturkar pada', 'bhadrawati', 'bhadwad', 'bhal gaon', 'bhandup east', 'bhandup west', 'bharat nagar', 'bhawani peth', 'bhayandar east', 'bhayandar west', 'bhayandarpada', 'bhegade aali', 'bhende layout', 'bhilarewadi', 'bhilgaon', 'bhivpuri', 'bhiwandi', 'bhokara', 'bhole nagar', 'bhosari', 'bhugaon', 'bhukum', 'bhuleshwar', 'bhusari colony right', 'bibwewadi', 'boisar', 'boisar west', 'bolinj naka', 'bopodi', 'borivali east', 'borivali west', 'byculla', 'chakan', 'chandansar', 'chandgavhan', 'chandivali', 'chandrakiran nagar', 'charholi budruk', 'charholi kurd', 'chatrapati nagar', 'chavani', 'chedda nagar', 'chembur', 'chembur east', 'chikhali', 'chinchbhuwan', 'chinchpada', 'chinchpada gaon', 'chinchwad', 'chunabhatti', 'cidco', 'colaba', 'dn nagar road', 'dabha ring road', 'dadar east', 'dadar west', 'dahisar', 'dahisar east', 'dahisar navi mumbai', 'dahisar west', 'dahivali', 'damat', 'dasak', 'datiwali gaon', 'dattavadi', 'deccan gymkhana', 'deepali nagar', 'dehu', 'deo nagar', 'deolali camp', 'deolali gaon', 'deonar', 'desale pada', 'devi nagar', 'devrung', 'dhamote', 'dhankawadi police station road', 'dhanori', 'dhanorilohegaon road', 'dhansar', 'dhantoli', 'dharampeth', 'dharavi', 'dhayari', 'dhayari phata', 'digha', 'dighe', 'dighi', 'dighori', 'dighori road', 'diva', 'diva gaon', 'dixit nagar', 'dombivali', 'dombivali east', 'dombivli (west)', 'dronagiri', 'dundare', 'dwarka', 'dwarli gaon', 'eastern express highway vikhroli', 'erandwane', 'fatherwadi', 'fort', 'four bungalows', 'friends colony', 'fursungi', 'fursungi bhekrai road', 'fursungi gaon', 'gtb nagar', 'gahunje', 'gaikwad vasti', 'gajanan colony', 'gajanan nagar', 'gandhi nagar', 'gauripada', 'ghansoli', 'ghatkopar', 'ghatkopar east', 'ghatkopar east pant nagar', 'ghatkopar west', 'ghodbunder road', 'ghodbunder thane west', 'ghogali', 'girgaon', 'godhani road', 'godhni', 'gopal nagar', 'goregaon east', 'goregaon west', 'gorewada', 'gotal pajri', 'govandi', 'govind nagar', 'greater khanda', 'gultekdi', 'hadapsar', 'haji malang road kalyan e', 'handewadi', 'haranwali', 'hariyali', 'hazari pahad nagpur', 'hazaripahad', 'hendre pada', 'hindustan colony', 'hingna', 'hinjewadi', 'hiranandani estates', 'hudkeshwar road', 'it park road', 'indira nagar', 'indraprastha nagar swavalambi nagar', 'indraprasthnagar', 'irani wadi', 'j b nagar', 'jacob circle', 'jadhavwadi', 'jafar nagar', 'jai prakash nagar', 'jaitala', 'jaitala road', 'jambhe', 'jambhul', 'jamtha', 'janki nagar', 'jawahar nagar', 'jogeshwari east', 'jogeshwari west', 'juhu', 'juhu vile parle development scheme', 'jui', 'juinagar', 'juinagar west', 'junnar', 'kt nagar', 'kachimet', 'kalamboli', 'kalher', 'kalpataru nagar', 'kalwa', 'kalyan', 'kalyan east', 'kalyan west', 'kalyani nagar', 'kalyani nagar annexe', 'kamala', 'kamatwade', 'kamothe', 'kamptee', 'kamptee road', 'kamshet', 'kandivali east', 'kandivali west', 'kanjurmarg', 'kanjurmarg east', 'kankavli', 'karanjade', 'karjat', 'karve nagar', 'kashele', 'kasheli', 'kashimira', 'katol road', 'katraj', 'katraj ambegaon bk road', 'kedgaon', 'keshav nagar', 'keshav nagar road', 'kewale', 'khalapur', 'khamla', 'khanda colony', 'khandala', 'khandeshwar', 'khapri', 'khar', 'khar west', 'kharadi', 'kharbi', 'kharbi road', 'khardi', 'khare town', 'kharghar', 'khed', 'khopoli', 'kiwale', 'kolsewadi', 'kolshet road thane west', 'kondhwa', 'kondhwa budruk', 'kongaon', 'kopargaon', 'koper khairane', 'koproli', 'koradi road', 'koregaon park', 'kothrud', 'kudal', 'kudale baug', 'kurla', 'kurla east', 'kurla west', 'lavale', 'laxminagar', 'lohegaon', 'lok dhara', 'loni kalbhor', 'lower parel', 'mae campus', 'matunga west', 'midc ambad', 'magarpatta', 'mahad', 'mahal', 'mahalaxmi', 'mahalunge', 'mahalunge ingale', 'mahavir nagar', 'mahim', 'majiwada thane', 'makhmalabad', 'malabar hill', 'malad east', 'malad west', 'mamdapur', 'mamurdi', 'manewada', 'manewada besa ghogli road', 'manewada ring road', 'manewada road', 'mangalmurti nagar', 'manish nagar', 'manish nagar main', 'manjari', 'manjari budruk', 'manjari khurd', 'mankapur', 'mankapur road', 'marine lines', 'matunga', 'mazagaon', 'mazgaon', 'mhatre nagar', 'mihan', 'mira road', 'mira road east', 'mira road west', 'model colony', 'mohammed ali rd', 'mohammed wadi', 'moshi', 'moshi pradhikaran pimprichinchwad', 'mukund nagar', 'mulshi', 'mulund', 'mulund east', 'mulund west', 'mumbai central', 'mumbra', 'mundhwa', 'murbad', 'nibm annex mohammadwadi', 'nibm road', 'nagaon', 'nahur east', 'naigaon east', 'naigaon west', 'nala sopara', 'nalasopara east', 'nalasopara west', 'nanded', 'nandivali gaon', 'napeansea road', 'nara', 'narendra nagar', 'narhe', 'nari road', 'nashik road', 'navade', 'nehru nagar kurla', 'neral', 'neral matheran road', 'nere', 'nerul', 'new balaji nagar', 'new d n nagar dnnagar', 'new kalyani nagar', 'new khapri', 'new manish nagar', 'new panvel east', 'new sangavi', 'nigdi', 'nilje gaon', 'north shivaji nagar', 'old nasheman colony', 'omkar nagar', 'omkar nagar road', 'oros', 'oshiwara', 'oshiwara garden road', 'outer ring road', 'padegaon', 'palaspe phata near by marathon nexzone project', 'palava', 'pale gaon', 'palghar', 'palm beach road vashi', 'panchak', 'panchavati', 'panjari', 'pant nagar', 'panvel', 'parel', 'parhur', 'parsodi', 'parvati darshan', 'pashan', 'pathardi gaon', 'pathardi phata', 'paud rd', 'pawne', 'phase 3', 'pimple gurav', 'pimple nilakh', 'pimple saudagar', 'pimpri', 'pipla', 'pirangut', 'pisavli village', 'pisoli', 'pokhran road no 2', 'porwal road', 'powai', 'prabhadevi', 'prabhat colony', 'pratap nagar', 'punawale', 'pune mumbai highway', 'pune satara road', 'rabale', 'rahatani', 'rahatgaon', 'rajapeth', 'rajawadi colony', 'ram nagar', 'ram wadi', 'ramdaspeth', 'rameshwari', 'ranjanpada uran panvel road', 'rasayani', 'ravet', 'residential flat virar west', 'rohinjan', 'shahad station', 'sadapur', 'sadar bazar', 'sadashiv peth', 'sagaon', 'sahakar nagar', 'sahkar nagar', 'saki vihar road', 'samata nagar thakur village', 'sangamvadi', 'sanmarga nagar', 'sanpada', 'santacruz east', 'santacruz west', 'santosh nagar', 'saphale', 'sarsole', 'satpati', 'savitribai phule nagar', 'seawoods', 'sector 21 kamothe', 'sector 37', 'seminary hills', 'sewri', 'shahapur', 'shahnoorwadi', 'shambhu nagar', 'shankar kalat nagar', 'shankar nagar', 'sharanpur', 'shastri nagar', 'shegaon rahatgaon road', 'shelu', 'shendra midc', 'shewalewadi', 'shil phata', 'shilottar raichur', 'shilphata diva naka', 'shilphata road thane', 'shirgaon', 'shirwal', 'shiv shakti nagar', 'shivaji colony', 'shivaji nagar', 'shivaji park', 'shivane', 'shivtirthanagar', 'shri krishna nagar', 'shrirampur', 'shukrawar peth', 'sindhi society chembur', 'sinhgad road', 'sion', 'somalwada', 'somalwada besa road', 'somatane', 'somwar peth', 'sonegaon', 'sopan baug', 'state bank colony', 'sunarwadi', 'surendra nagar', 'sus', 'suvarna krida', 'suyog nagar', 'swaraj nagar', 'swargate', 'swawlambi nagar', 'syndicate', 'talegaon', 'talegaon dabhade', 'talegaon dhamdhere', 'taljai road', 'taloja', 'taloja phase 2', 'taloje', 'tardeo', 'tarwala nagar', 'tathawade', 'telecom nagar', 'thakur village', 'thakurli', 'thane bhiwandi road', 'thane east', 'thane west', 'thane diva', 'themghar', 'thergaon', 'tilak nagar', 'tilak nagar mumbai', 'tingre nagar', 'titwala', 'titwala east', 'trimurti nagar', 'tukum', 'uday nagar', 'uday nagar ring road', 'ujwal nagar', 'ulhasnagar', 'ulwe', 'umred road', 'umroli', 'undri', 'uran', 'uruli devachi', 'uruli kanchan', 'usarghar', 'usatane', 'uttan', 'vadgaon', 'vadgaon budruk', 'vadgaon sheri', 'vallabh baug lane', 'vangani', 'vartak nagar', 'vasai', 'vasai west', 'vasai east', 'vasant vihar', 'vashi', 'vashi kopar khairane road', 'vasind', 'vayusena nagar', 'veera desai road', 'vevoor', 'vichumbe', 'vidya vihar east', 'vikhroli', 'vikhroli west', 'vikroli east', 'ville parle east', 'ville parle west', 'viman nagar', 'vindhane', 'virar', 'virar east', 'virar west', 'vishrambag', 'vishrantwadi', 'vitthal nagar 1', 'vyankatesh nagar', 'wada', 'wadala', 'wadala east', 'wadgaon sheri', 'wadi road', 'wagdara', 'wagholi', 'wakad', 'wakad pune', 'wanowrie', 'warai', 'wardha road', 'warje', 'wathoda', 'wavandhal', 'worli', 'yerawada', 'yewalewadi', 'yogidham', 'zalta', 'zingabai takli', 'bhekarai nagar', 'dombivli west', 'harigram', 'hingne khurd', 'kandivali', 'kanjurmarg west', 'karanjade panvel', 'karjat near to railway station', 'kasaradavali thane west', 'khadakpada', 'kopri', 'matunga east', 'mumbai', 'nagpur', 'nallasopara w', 'new panvel navi mumbai', 'pannase layout', 'sinhagad road', 'thakur village kandivali east', 'ulhasnagar 4', 'vasant vihar thane west', 'versova', 'vile parle west'])


#st.button('Predict House Price', onclick=predict)
if st.button("Predict House Price"):
        output = get_estimated_price(locality_name,sqft,bhk)
        st.success(get_estimated_price(locality_name,sqft,bhk))



html_temp = """
    <div style="background:#025246 ;padding:2px">
    <h4 style="color:white;text-align:center;"> Created by Nihar Jagdish Panchal. </h4>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html = True)