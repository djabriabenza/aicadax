# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 00:48:46 2025

@author: djabr
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# تحميل البيانات
df = pd.read_csv("D:\Datasets/course.csv")

# تصفية القيم الصحيحة فقط في عمود Enroled (0 و 1 فقط)
df = df[df['Enroled'].isin([0, 1])]

# تحويل التواريخ إلى تنسيق زمني
df['Timecreated'] = pd.to_datetime(df['Timecreated'], errors='coerce')
df['Timemodified'] = pd.to_datetime(df['Timemodified'], errors='coerce')

# استخراج ميزات من التواريخ
df['created_year'] = df['Timecreated'].dt.year
df['created_month'] = df['Timecreated'].dt.month
df['created_day'] = df['Timecreated'].dt.day

df['modified_year'] = df['Timemodified'].dt.year
df['modified_month'] = df['Timemodified'].dt.month
df['modified_day'] = df['Timemodified'].dt.day

# التعامل مع النص في عمود fullname باستخدام TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
fullname_features = vectorizer.fit_transform(df['fullname'].astype(str)).toarray()

# اختيار الميزات الزمنية
time_features = df[['created_year', 'created_month', 'created_day',
                    'modified_year', 'modified_month', 'modified_day',
                    'DateDifference']].fillna(0).values

# دمج الميزات
X = np.hstack((fullname_features, time_features))
y = df['Enroled']

# تقسيم البيانات لتدريب النموذج
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# طباعة تقرير الأداء
print(classification_report(y_test, y_pred))

