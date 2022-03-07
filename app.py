import streamlit as st
import pandas as pd 
from transformers import *
import numpy as np 
import torch
import requests
from bs4 import BeautifulSoup as bs
import os
import pandas as pd



if __name__ == "__main__":
    
    st.title("News classifier [international, scitech, sports, magazine]") # title of webpage
    
    user_input = st.text_input("article-link") # User input

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = 100
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # tokenizer


    model =  DistilBertForSequenceClassification.from_pretrained("/home/utkarsh_pratiush/others/kure/distilbert_model_news").to(device)


    soup = bs(requests.get(user_input).__dict__['_content'], "html5lib")
    article = ''
    for p in soup.find_all('p'): # looping through paragraphs in article
                    p.get_text()
                    article = article + p.get_text()
                
                

    tok = tokenizer(article,max_length=max_length,pad_to_max_length=True)
    test_input_ids = tok['input_ids']
    test_token_type_ids = tok['token_type_ids']
    test_attention_masks = tok['attention_mask']
    test_inputs = torch.tensor(test_input_ids).to(device).resize_(1,100)
    #test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks).to(device).resize_(1,100)
    test_token_types = torch.tensor(test_token_type_ids).to(device).resize_(1,100)
    
    

    model.eval()
    outs = model.forward(test_inputs, attention_mask=test_masks)
    probability = torch.sigmoid(outs[0])
    st.write(probability)