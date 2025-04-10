# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 03:22:58 2025

@author: djabr
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("D:\Datasets/etud.csv")

import pandas as pd

# تحميل الملف
df = pd.read_csv("D:\\Datasets\\etud.csv")  # لاحظ استخدام \\ أو r"..." لتفادي مشاكل المسار

# عرض أول 5 صفوف
print("أول 5 صفوف من البيانات:")
print(df.head())

# عرض معلومات عامة عن الأعمدة والبيانات
print("\nمعلومات عامة عن البيانات:")
print(df.info())

# عرض عدد الطلاب في كل مادة (Course)
print("\nعدد الطلاب في كل مادة:")
print(df['Course'].value_counts())

# عرض قائمة المواد الفريدة
print("\nقائمة المواد الفريدة:")
print(df['Course'].unique())


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# تحميل البيانات
df = pd.read_csv("D:\\Datasets\\etud.csv")

# إنشاء اسم كامل لتحديد كل طالب
df['FullName'] = df['Firstname'] + ' ' + df['Lastname']

# pivot table: الصفوف = الطلاب، الأعمدة = المواد، القيم = 1 إذا مسجل
pivot_df = pd.crosstab(df['FullName'], df['Course'])

# حساب تشابه الطلاب باستخدام cosine similarity
similarity_matrix = cosine_similarity(pivot_df)

# تحويل المصفوفة إلى DataFrame لسهولة التعامل
similarity_df = pd.DataFrame(similarity_matrix, index=pivot_df.index, columns=pivot_df.index)

# اختيار طالب معين (مثال)
target_student = pivot_df.index[0]  # ممكن تغييره لأي اسم من القائمة

# استخراج أكثر الطلاب تشابهًا
most_similar_students = similarity_df[target_student].sort_values(ascending=False)[1:6]  # استثناء نفسه

# استخراج الدورات التي أخذها الطالب
student_courses = set(df[df['FullName'] == target_student]['Course'])

# توصيات: دورات أخذها الطلاب المتشابهون ولم يأخذها الطالب الحالي
recommended_courses = set()
for similar_student in most_similar_students.index:
    courses = set(df[df['FullName'] == similar_student]['Course'])
    recommended_courses.update(courses - student_courses)

print(f"الدورات المقترحة للطالب {target_student}:")
print(recommended_courses)

import numpy as np

# إضافة مستوى أكاديمي عشوائي
levels = ['سنة أولى', 'سنة ثانية', 'ماجستير']
df['Level'] = np.random.choice(levels, size=len(df))

# إضافة علامات عشوائية (بين 10 و 20)
df['Grade'] = np.round(np.random.uniform(10, 20, size=len(df)), 2)

# تقييم المادة من طرف الطالب (1 إلى 5 نجوم)
df['Rating'] = np.random.randint(1, 6, size=len(df))

# التحقق من الأعمدة الجديدة
print(df.head())


from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# pivot: كل طالب = صف، وكل دورة = عمود (تأخذ القيم العلامة)
pivot_grades = df.pivot_table(index='FullName', columns='Course', values='Grade', fill_value=0)

# pivot: التقييمات
pivot_ratings = df.pivot_table(index='FullName', columns='Course', values='Rating', fill_value=0)

# دمج الدرجات + التقييمات
combined_features = pd.concat([pivot_grades, pivot_ratings], axis=1)

# تحويل المستوى إلى أرقام
level_mapping = {'سنة أولى': 1, 'سنة ثانية': 2, 'ماجستير': 3}
student_levels = df[['FullName', 'Level']].drop_duplicates().set_index('FullName')
student_levels['Level_Num'] = student_levels['Level'].map(level_mapping)

# دمج المستوى
combined_features = combined_features.merge(student_levels['Level_Num'], left_index=True, right_index=True)

# توحيد القيم (Normalization)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(combined_features)

# حساب التشابه
similarity_matrix = cosine_similarity(features_scaled)
similarity_df = pd.DataFrame(similarity_matrix, index=combined_features.index, columns=combined_features.index)

# توصية لدراسة طالب معين
target_student = combined_features.index[0]

# الطلاب الأكثر تشابهًا
similar_students = similarity_df[target_student].sort_values(ascending=False)[1:6]

# استخراج الدورات التي لم يأخذها الطالب
student_courses = set(df[df['FullName'] == target_student]['Course'])
recommended_courses = set()

for sim_student in similar_students.index:
    courses = set(df[df['FullName'] == sim_student]['Course'])
    recommended_courses.update(courses - student_courses)

print(f"🔮 الدورات المقترحة للطالب {target_student}:")
for course in recommended_courses:
    print("✅", course)