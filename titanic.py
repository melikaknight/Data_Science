import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

# ابتدا فایل مورد نظر داخل یک دیتا فریم پانداز بارگذاری می‌شود
train_set = pd.read_csv("train.csv")
# ستون های بلیط، نام و کابین را از آن حذف می‌کنیم
train_set = train_set.drop(['Ticket','Name','Cabin'],axis=1)

# به این ترتیب مدل باید بر مبنای سایر ستون ها، ستون مربوط به زنده ماندن را پیش بینی کند
X = train_set.loc[:, train_set.columns != 'Survived']
# ستون مربوط به زنده ماندن که همان لیبل پیش بینی است را به صورت بولین ذخیره می‌کنیم
y = train_set['Survived'].astype(bool)

# یک ترانسفورمر تعریف می‌کنیم که بر روی داده های ستون به جنسیت و محل سوار شدن که از نوع دسته ای هستند، با استفاده از انکودر به عدد تبدیل شوند
# همچنین مقادیر خالی و نامشخص در ستون مربوط به سن نیز با استفاده از میانگین داده‌های این ستون پر می‌شود
transformer = ColumnTransformer([
    ('OneHotEncoder', OneHotEncoder(), ['Sex','Embarked']),
    ('SimpleImputer', SimpleImputer(), ['Age']),
],remainder='passthrough')
# یک پایپلاین تعریف می‌کنیم که عملیات تعریف شده داخل آن به ترتیب بر روی داده اعمال می‌شود
# در این پایپلاین، ابتدا ترانسفوری که در بخش بالا ساختیم را بر روی داده اعمال کند.
# سپس با استفاده از مین مکس اسکیلر، تمامی داده ها را به بازه ۰ تا ۱ تبدیل می‌کنیم.
# در نهایت با اعمال یک پی‌سی‌ای، کاهش بعد انجام می‌دهیم و تعداد بعد ها را به ۵ بعد کاهش می‌دهیم، به این ترتیب داده‌ها ۵ ستون می‌شود
pipe = Pipeline([
    ('transformer', transformer),
    ('minmaxscalar', MinMaxScaler()),
    ('pca', PCA(n_components=5))
])
# داده های ورودی را به پایپلاین تعریف شده می‌دهیم تا پیش‌پردازش تعریف شده بر روی آن اعمال شود
X_preprocessed = pipe.fit_transform(X)

# در اینجا الگوریتم نزدیک ترین همسایه پیاده سازی شده است که کار پیش بینی را انجام می‌دهد
# این تابع دیتای آموزشی و لیبل های آن را دریافت می‌کند و با دریافت دیتای ارزیابی، لیبل‌های آن را پیش بینی می‌کند
# همچنین متغیر تعداد همسایه را نیز به عنوان ورودی دریافت می‌کند
def KNN_predict(X_tr,y_tr,X_vl,neighbors):
    # در اینجا لیبل های داده های آموزش را به یک لیست تبدیل می‌کنیم
    listed_y_tr = list(y_tr)
    # یک آرایه برای ذخیره نتیجه‌ی پیش بینی ها ایجاد می‌کنیم
    predicts = []
    # به ازای هر سطر از داده ی ارزیابی
    for vl_item in X_vl:
        # فاصله‌ی داده ارزیابی از هر داده‌های آموزشی به همراه لیبل آن داده ی آموزشی در این آرایه ذخیره می‌شود
        distances = []
        # به ازای هر سطر از داده ی آموزشی
        for i,tr_item in enumerate(X_tr):
            # .فاصله‌ی داده ارزیابی از آن داده آموزشی به همراه لیبل آن در آرایه اضافه می‌شود. معیار فاصله را، فاصله اقلیدسی در نظر گرفتیم
            distances.append((np.linalg.norm(vl_item-tr_item),listed_y_tr[i]))
        # با مرتب سازی، نزدیک ترین همسایه ها را بر کمترین فاصله به تعدادی که در ورودی تابع مشخص شده است جدا می‌کنیم
        nearest_neighbors = sorted(distances, key=lambda x:x[0])[:neighbors]
        # لیبل های مربوط به این نزدیک ترین همسایه ها را جدا می‌کنیم تا بر اساس آن پیش بینی را انجام دهیم
        nearest_neighbors_labels = [x[1] for x in nearest_neighbors]
        # اگر تعداد لیبل های مثبت بیشتر از نصف تعداد عمسایه های بود، پیش بینی ما مثبت است، در غیر این صورت پیش بینی منفی است
        if sum(nearest_neighbors_labels) > neighbors/2:
            predicts.append(True)
        else:
            predicts.append(False)
    return predicts


# با استفاده از کی‌فولد، کل داده‌ها را به ۵ قسمت تقسیم می‌کنیم، و هر بار یک قسمت را به عنوان داده‌ی ارزیابی و ۴ قسمت دیگر را به عنوان داده آموزشی در نظر می‌گیریم
# به این ترتیب ۵ بار عملیات یادگیری صورت می‌گیرد تا بتوانیم دقت میانگین را محاسبه کنیم
kf = KFold(n_splits=5)

# در هر مرحله دقت به دست آمده در این لیست ذخیره می‌شود
all_folds_accuracy = []

# در این حلقه، با توجه به کی‌فولد تعریف شده، ۵ بار یاد‌گیری را انجام می‌دهیم
for train_index, test_index in kf.split(X):
    # در این قسمت داده‌های آموزشی و ارزیابی و لیبل‌های مربوط به آن در هر مرحله مشخص می‌گردد
    X_train, X_valid = X_preprocessed[train_index], X_preprocessed[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # با استفاده از تابعی که برای الگوریتم نزدیک ترین همسایه ارائه کردیم، یادگیری را انجام می‌دهیم و برای داده‌های ارزیابی، لیبل‌ها را پیش بینی می‌کنیم
    # بهترین تنظیم به دست آمده برابر با ۲۵ است که ۲۵ همسایه نزدیک را در نظر می‌گیرد.
    prediction = KNN_predict(X_train,y_train,X_valid,25)
    # پیش بینی ‌های به دست آمده را با مقادیل واقعی لیبل های ارزیابی مقایسه می‌کنیم تا دقت محاسبه شود
    accuracy = accuracy_score(y_valid, prediction)
    # دقت به دست آمده در این مرحله را در آرایه مد نظر اضافه می‌کنیم
    all_folds_accuracy.append(accuracy)
    
# در اینجا میانگین دقت ۵ مرحله محاسبه و چاپ می‌شود
print("Average accuracy: {}%".format(round(np.average(all_folds_accuracy)*100,2)))



