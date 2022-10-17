#!/usr/bin/env python
# coding: utf-8

# # EnvBayes Logo
# The only self-respecting way for a Python person to make a logo is with Scipy, Numpy, and Matplotlib.  

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import numpy as np
sns.set_palette("Greens_r")


# In[2]:


fig, ax = plt.subplots(figsize=(3,1))
x = np.linspace(-10, 10, 1000)
kappas =  [1, 1/2]
offsets = [0, 1.1]
coeffs  = [.3, .3]
heights = [ 1, .8]
for kappa, offset, coeff, height in zip(kappas, offsets, coeffs, heights):
    rv = st.laplace_asymmetric(kappa)
    ax.plot(x, height * rv.pdf(coeff * x + offset), lw=3)
ax.text(4.6, 0.3, 'EnvBayes')
plt.axis('off')
plt.tight_layout()
plt.savefig('logo', dpi=200)


# The [asymmetric laplace distribution](https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution) is two exponential distributions standing back-to-back.  In Bayesian statistics, it can be used in quantile regression.  
# 
# The pair of asymetric laplace distributions in the logo remind me of a stylized version of Mt. Rainier and Little Tahoma in my home state of Washington.   

# In[ ]:




