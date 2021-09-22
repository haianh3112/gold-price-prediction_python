# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from datetime import datetime, date
import yfinance as yf
import streamlit as st
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


gold_data = pd.read_excel("")
gold_data.head()
