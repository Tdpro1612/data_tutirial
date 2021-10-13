## step one : get data

```
from google.colab import drive
drive.mount("/content/drive")
```
```
file_name = "/content/kaggle.json"
with open(file_name, 'r') as f:
    document =  js.loads(f.read())
    
print(document)
#{'key': 'c00046e4b093c67705ca6d402f00aa31', 'username': 'nguynnguynkhoa'}

os.environ['KAGGLE_USERNAME'] = document['username']
os.environ['KAGGLE_KEY'] = document['key']

#get API
```
```
!kaggle datasets download -d uciml/adult-census-income
```
```
!unzip /content/adult-census-income.zip
```
```
df = pd.read_csv("/content/adult.csv")
```
# step 2 : Problem description
```
df.info()
```
df.head()
```
```
df['workclass'].value_counts()
```
```
df['education'].value_counts()
```
