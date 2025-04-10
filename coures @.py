# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 02:08:50 2025

@author: djabr
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# تحميل البيانات من المسار الصحيح
df = pd.read_csv("D:\Datasets/da2022.csv")

# تحويل الأيام من نص إلى أرقام (مثلاً "1st" تصبح 1)
df["Day_num"] = df["Day"].str.replace("st|nd|rd|th", "", regex=True).astype(int)

# تجهيز البيانات للنموذج
X = df[["Day_num"]]  # المدخلات: رقم اليوم
y = df["DistinctUserLogins"]  # المخرجات: عدد تسجيلات الدخول

# إنشاء وتدريب نموذج الانحدار الخطي
model = LinearRegression()
model.fit(X, y)

# التنبؤ بعدد تسجيلات الدخول لكل يوم من 1 إلى 31
days_range = np.arange(1, 32).reshape(-1, 1)
predictions = model.predict(days_range)

# رسم النتائج
plt.figure(figsize=(10, 6))
plt.scatter(df["Day_num"], y, label="Actual Logins", color='blue')        # البيانات الأصلية
plt.plot(days_range, predictions, label="Predicted Trend", linestyle='--', color='red')  # الاتجاه المتوقع
plt.xlabel("Day of the Month")
plt.ylabel("Distinct User Logins")
plt.title("Predicted vs Actual User Logins")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

