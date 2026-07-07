'''
In this file we are going to load the data from text file using
langchain Framework
'''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import langchain
import langchain_community
from langchain_community.document_loaders import TextLoader

sol = TextLoader("./../data_location/credit_card.txt")

result = sol.load()

for i in result:
    print(i.page_content)
