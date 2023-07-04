#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig


# In[2]:


#set the model
model_name = 't5-small'


# In[3]:


#Tokenize and load the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# In[4]:


#sets the maximum number of tokens the generated output can contain
config = GenerationConfig(max_new_tokens=200)


# In[5]:


#It iterates over each line of input until there is no more input available

for line in sys.stdin:
    tokens = tokenizer(line, return_tensors="pt")
    outputs = model.generate(**tokens, generation_config=config)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


# In[ ]:




