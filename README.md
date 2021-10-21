# IMDB-sentiment-analysis

### 📝Description
---
In this project, we are going to implement an End-to-End and basic Natural Language Processing project which is sentiment analysis on IMDB movie reviews. In this project our purpose is to analyse and predict the reviewer's feeling about a movie which is either positive or negative. The dataset which is used in this project is `IMDB Dataset of 50K Movie Reviews` which you can download it from this [link](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The dataset is not completely clean and in order to clean this dataset various text preprocessing steps were used such as `Stemming`, `Stop Words Removal`, `Regular Expressions`, etc. The cleaned dataset is available in this repository as `cleaned_data.csv`.

In order to represent features, two methods were used which are `TF-IDF` and `Word Embedding`. Word Embedding has less ccomputational complexity due to less sparsity. So this method is used in production.

Two models were used, Naive Bayes classifier which is based on bayesian theorem and predicts the label based on features/words related to that specific feature. The other model is 1D CNN. Both models yield the same accuracy and result(`~%85`). Models are available in this repo and can be accessed by `models` folder.

### 🖥 Installation
---
The Code is written in Python 3.7.5. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensure you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:
```
git@github.com:git@github.com:Kasra1377/document-invoice.git
```
or
```
git clone https://github.com/Kasra1377/document-invoice.git
```


### 📉Project Results
---
The model has been created and put into web application and you can see the performance and the output of the model down below:
<p align="center">
  <img width="900" height="500" src="demo/SQL-database.PNG">
</p>
 
### 🛠Technologies Used
---
**IDE**:  ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)


**Language(s):**  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

**Libraries:**  ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  ![MySQL](https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white)
    ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

**Database(s):**  ![MySQL](https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white)

### ❌Bugs & Issues
---
If you ever encountered any bugs or any technical issues in this projects you can report it by `issues` section of this repository or you can contact me by my email address. 

### 👥Contributers
---
Kasra1377
