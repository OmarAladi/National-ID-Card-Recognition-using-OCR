#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img_path = "new1.jpg"
config = "--psm 4"

"""
Note:
National ID Card images must be inserted horizontally
"""


# # 1) Load the image in grayscale

# In[2]:


img = cv2.imread(img_path, 0)

plt.figure(figsize=(7,5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("color image")


# # 2) Resizeg the image

# In[3]:


img_resize = cv2.resize(img, (None), fx = 0.5, fy = 0.5)

plt.figure(figsize=(7,5))
plt.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)), plt.title("resized color image")


# # 3) Thresholding

# ## 3.1) Convert image to black and white using Binart Inverse Thresholding 

# In[4]:


_, img_thresh1 = cv2.threshold(img_resize, 70, 140, cv2.THRESH_BINARY_INV)

plt.figure(figsize=(7,5))
plt.imshow(img_thresh1, "gray")


# ## 3.2) Convert image to black and white using Adaptive Thresholding (Gaussian Mean Thresholding)

# In[5]:


img_thresh2 = cv2.adaptiveThreshold(img_resize, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 281, 81)

plt.figure(figsize=(7,5))
plt.imshow(img_thresh2, "gray")


# # 4) Morphological Transformations

# # 4.2) Dilation

# In[6]:


kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(img_thresh1, kernel, iterations = 1)

plt.figure(figsize=(7,5))
plt.imshow(dilation, "gray"), plt.title("Dilation image")


# ## 4.1) Opening

# In[7]:


kernel = np.ones((4,4),np.uint8)
opening_img = cv2.morphologyEx(img_thresh2, cv2.MORPH_OPEN, kernel)
plt.imshow(opening_img, "gray"), plt.title("opening image")


# # 5) Apply Tesseract OCR

# In[8]:


text1 = pytesseract.image_to_string(dilation, config=config,lang="ara")
print(text1)

print("-----------------------------------------------------------------------------")


text2 = pytesseract.image_to_string(opening_img, config=config,lang="ara")
print(text2)


# In[9]:


print(text2.splitlines())


# In[10]:


columns_lst = [["first name", "last name", "Address", "Governorate"],
               [text2.splitlines()[0], text2.splitlines()[1], text2.splitlines()[2], text2.splitlines()[3]]]

df = pd.DataFrame(columns_lst, columns=["first name", "last name", "Address", "Governorate"])
df = df.drop(0)
df

