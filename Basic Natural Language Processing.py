#Natural Language Processing
'''
Natural Language Processing refers to the application of 
computational techniques to the analysis and synthesis of natural 
language and speech for a machine.
''' 

#Bag of Words is one of the most popular algorithm used in Natural Language Processing

'''
Natural Language Toolkit library is the library which is provided
by scikit learn to do NLP on datasets.
We can also use Tensorflow for this purpose.

'''
import pandas as pd #importing pandas to read dataset.

#using a delimiter to read the tab space present in the file
dataset = pd.read_csv('C:/Users/abc/Desktop/Machine Learning/Restaurant_Reviews.tsv',delimiter='\t')
review = dataset['Review']

#Using Regular expressions to read only the text data inside the file
import re 

#creating a fucntion to use regular expressions to convert the characters to string format
def preprocessed_text(review):
    cleaned_text = re.sub(r"[^a-z A-Z 0-9\s]", " ",review).lower()
    return cleaned_text

#reading and preprocessing the data
review = []
labels = []

tsv_file_path = ":/Users/abc/Desktop/Machine Learning/Restaurant_Reviews.tsv',delimiter='\t'"

with open(tsv_file_path, 'r', encoding="UTF-8") as tsv_file:
    tsv_reader = dataset(tsv_file, delimiter = '\t')
    next(tsv_reader)#skip the header row if it alreadt exists
    for row in tsv_reader:
        if len(row) >= 2:  #Ensuring the row has at least 2 columns
            review_text = row[0]
            liked = int(row)
            cleaned_review = preprocessed_text(review_text)
            review.append(cleaned_review)
            labels.append(liked)
            
#Using NLTK library to apply NLP to the dataset
import nltk
nltk.download('stopwords')

#converting the review to lowercase and spilting 
review = review.split()
notstopwords = ['not', 'dont'] 

#importing corpus from nltk
from nltk.corpus import stopwords

#corpus = a collection of authentic text or audio organized into dataset
corpus = []
mybag = []

for data in review:
    if(data not in stopwords.words('english')):
        mybag.append(data)

mystembag = []

#Importing the PorterStemmer to find the root words of the words
from nltk.stem.porter import PorterStemmer
for data in mybag:
    ps = PorterStemmer()
    stemdata = ps.stemdata(data)
    if stemdata not in mystembag:
        mystembag.append(stemdata)
        
#joining the review and mystembag
review = ' '.join(mystembag)
corpus.append(review)

#for preprocessing importing CountVectorizer from feature extraction
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(review_text)
