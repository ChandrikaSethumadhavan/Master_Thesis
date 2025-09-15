import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO




data_path = r"C:\Users\chand\Documents\GitHub\Thesis\2.5mM EDA\ntotal_output_RT_2.5mM.csv"
df_2 = r"C:\Users\chand\Documents\GitHub\Thesis\2.5mM EDA\henrys_constants_RT.csv"


df1 = pd.read_csv(data_path)
df2 = pd.read_csv(df_2)
combined_df = pd.concat([df1, df2], axis =1)
combined_df.to_csv("ntot_RT_2.5.csv", index=False)



