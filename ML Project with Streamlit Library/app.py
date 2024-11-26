from PIL import Image
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix

class App:
    def __init__(self):
        self.df = None
        self.classifier_name = None
        self.load_streamlit_page()
        self.model = None
        self.X, self.y = None, None


    def load_streamlit_page(self):
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv'])
        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            (' ','SVM', 'KNN', 'Naïve Bayes',)
        )
        if uploaded_file is not None:
            st.title('Breast Cancer Model With Different Classifiers')
            self.df = pd.read_csv(uploaded_file)
            st.header("First 10 Rows of Data :")
            st.write(self.df.head(10))

            st.header("Columns of Data:")
            st.write(self.df.columns)

            self.df.drop(['id', 'Unnamed: 32'], inplace=True, axis=1)
            self.df['diagnosis'] = self.df['diagnosis'].replace({'M': 1, 'B': 0})

            fig1 = self.plot_correlation_matrix()
            fig2 = self.plot_scatter()
            st.write(f'Dataset shape : ',self.df.shape)
            st.header("Correlation Matrix :")
            st.pyplot(fig1)
            st.header("Scatter Plot :")
            st.pyplot(fig2)
            """
            corr_matrix = self.df.corr().abs()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))   #Korelasyona göre column'ları azaltınca accuracy değerleri azaldı.
            tri_df = corr_matrix.mask(mask)

            to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]

            self.df = self.df.drop(to_drop, axis=1)
            """
            X = self.df.drop('diagnosis', axis=1)
            y = self.df['diagnosis']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
            self.df = pd.concat([y, X_scaled_df], axis=1)

            self.X = X_scaled_df
            self.y = y
            st.header("Last 10 Rows After Scaling :")
            st.write(self.df.tail(10))

            self.select_model()
            self.train_model()


        else:
           st.title("Enter Your Dataset")
           st.info("You have to enter your dataset!")
           image_path = 'enterdata.png'
           image = Image.open(image_path)
           new_size = (550, 550)
           resized_image = image.resize(new_size)
           st.image(resized_image)

    def plot_correlation_matrix(self):
        corr = self.df.corr()
        plt.figure(figsize=(20,12))
        sns.heatmap(corr, annot=True, fmt=".2f",cmap="crest",linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        return plt.gcf()

    def plot_scatter(self):
        malignant = self.df[self.df['diagnosis'] == 1]
        benign = self.df[self.df['diagnosis'] == 0]

        plt.figure(figsize=(7, 6))
        sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant,label='Malignant', color='blue', alpha=0.6, sizes=(100, 200))
        sns.scatterplot(x='radius_mean', y='texture_mean', data=benign,label='Benign', color='green', alpha=0.6,sizes=(100, 200))
        plt.title('Radius Mean vs Texture Mean by Diagnosis')
        plt.xlabel('Radius Mean')
        plt.ylabel('Texture Mean')
        plt.legend()
        return plt.gcf()

    def select_model(self):
        if self.classifier_name == 'SVM':
            parameters = {
                'C': [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20],
                'gamma': [0.0001, 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            self.model = GridSearchCV(SVC(), parameters, cv=5, scoring='accuracy')

        elif self.classifier_name == 'KNN':
            parameters = {
                'n_neighbors': list(range(1, 20)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            self.model = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, scoring='accuracy',n_jobs=-1)
        elif self.classifier_name =='Naïve Bayes':
            self.model = GaussianNB()
        else:
            self.model = None


    def train_model(self):
        if self.model is not None:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            if hasattr(self.model, 'best_params_'):
                st.header(f'Best Parameters for {self.classifier_name}: ')
                best_params_df = pd.DataFrame([self.model.best_params_])
                html = best_params_df.to_html(index=False, justify='center')
                st.markdown(html, unsafe_allow_html=True)

                predictions = self.model.predict(X_test)
            else:
                predictions = self.model.predict(X_test)

            precision = precision_score(y_test, predictions, average='macro')
            recall = recall_score(y_test, predictions, average='macro')
            f1 = f1_score(y_test, predictions, average='macro')

            metrics = {
                ' ': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [accuracy_score(y_test, predictions), precision, recall, f1]
            }
            metrics_df = pd.DataFrame(metrics)
            st.header("Model Metrices : ")
            st.markdown(f'**Classifier Name:** {self.classifier_name}')
            st.table(metrics_df.set_index(' '))
            cm = confusion_matrix(y_test, self.model.predict(X_test))
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('True Values')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(['Benign (0)', 'Malignant (1)'])
            ax.yaxis.set_ticklabels(['Benign (0)', 'Malignant (1)'])

            st.header("Confusion Matrix")
            st.pyplot(fig)
            tn, fp, fn, tp = cm.ravel()
            confusion_metrics = {
                ' ': ['Actually Positive (1)', 'Actually Negative (0)'],
                'Predicted Positive (1)': [tp, fp],
                'Predicted Negative (0)': [fn, tn]
            }
            confusion_df = pd.DataFrame(confusion_metrics).set_index(' ')
            st.table(confusion_df)
        else:
            st.warning("The classifier is not selected")


