## step 1 : get data

```
from google.colab import drive
drive.mount("/content/drive")
```
Mounted at /content/drive
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
{'username': 'nguynnguynkhoa', 'key': '86d54991913d6b87880d297405c8173e'}
```
!kaggle datasets download -d uciml/adult-census-income
```
Downloading adult-census-income.zip to /content
  0% 0.00/450k [00:00<?, ?B/s]
100% 450k/450k [00:00<00:00, 67.7MB/s]
```
!unzip /content/adult-census-income.zip
```
Archive:  /content/adult-census-income.zip
  inflating: adult.csv               
```
df = pd.read_csv("/content/adult.csv")
```

# step 2 : Problem description

