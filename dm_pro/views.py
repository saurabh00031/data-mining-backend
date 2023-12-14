import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CSVFile
from django.http import HttpResponse,JsonResponse
from rest_framework.parsers import FileUploadParser,MultiPartParser,FormParser
from .models import CSVFile
from .serializers import CSVFileSerializer
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from django.http import FileResponse
import uuid
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score



class CSVFileUploadView(APIView):
    parser_classes = ( MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = CSVFileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




def home(request):
    return HttpResponse("Hello homepage")


def assignment1(request):
    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    


    data = pd.read_csv(node[0].file)
    mean_data=data.mean()
    median_data = data.median()
    mode_data = data.mode().iloc[0]  
    var_data = data.var()
    sd_data = data.std()


    print("mean",mean_data[0])
    print("median",median_data[0])
    print("mode",mode_data[0])
    print("variance",var_data[0])
    print("standard deviation",sd_data[0])
    

    my_data={
        "name":node[0].name,
        "mean":mean_data.to_dict(),
        "median":median_data.to_dict(),
        "mode":mode_data.to_dict(),
        "variance":var_data.to_dict(),
        "std":sd_data.to_dict()
    }
    return JsonResponse(my_data)






from io import StringIO

def assignment1_que2(request):
        node=CSVFile.objects.all()
        print(node[0].name)
        if len(node)==0 :
            return HttpResponse("No csv file in database !!")
        

        csv_file=node[0].file

        print("boundary 1")

        df = pd.read_csv(csv_file)

        df = df.drop('variety', axis=1)
  
        # Calculate various dispersion measures
        data = df.values.flatten()
        data = np.sort(data)

        # Range
        data_range = np.ptp(data)

        # Quartiles
        quartiles = np.percentile(data, [25, 50, 75])

        # Interquartile Range (IQR)
        iqr = quartiles[2] - quartiles[0]

        # Five-Number Summary
        five_number_summary = {
            "Minimum": np.min(data),
            "Q1 (25th Percentile)": quartiles[0],
            "Median (50th Percentile)": quartiles[1],
            "Q3 (75th Percentile)": quartiles[2],
            "Maximum": np.max(data)
        }

        csv_file.seek(0)  # Ensure the file pointer is at the beginning
        csv_data = csv_file.read().decode('utf-8')
        csv_buffer = StringIO(csv_data)

        column_name="sepal.length"
        column_name2="sepal.width"

        column_values=[]
        column_values2=[]

        csv_reader = csv.DictReader(csv_buffer)
        for row in csv_reader:
            if column_name in row:
                   column_values.append(row[column_name])

        csv_file.seek(0)  # Ensure the file pointer is at the beginning
        csv_data = csv_file.read().decode('utf-8')
        csv_buffer = StringIO(csv_data)
        csv_reader = csv.DictReader(csv_buffer)

        for row in csv_reader:
            if column_name2 in row:
                   column_values2.append(row[column_name2])
    
        

        result = {
            "Range": data_range,
            "Quartiles": quartiles.tolist(),
            "Interquartile": iqr,
            "Five": five_number_summary,
            "values":column_values,
            "values2":column_values2
        }

        return JsonResponse(result)


def assignment2(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    Attr1="sepal.length"
    Attr2="sepal.width"

    print("boundary 1")

    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    contingency_table = pd.crosstab(df[Attr1], df[Attr2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    alpha=0.7  # we have set that.....
    fl=0

    if p <= alpha:
        print(f"The p-value ({p}) is less than or equal to the significance level ({alpha}).")
        print("The selected attributes are correlated.")
        fl=1
    else:
        print(f"The p-value ({p}) is greater than the significance level ({alpha}).")
        print("The selected attributes are not correlated.")
        fl=0
    
    print(expected)
    correlation_coefficient = df[Attr1].corr(df[Attr2])
    covariance = df[Attr1].cov(df[Attr2])

    min_value = df[Attr1].min()
    max_value = df[Attr1].max()

    df[Attr2] = (df[Attr1] - min_value) / (max_value - min_value)

    mean = df[Attr1].mean()
    std_dev = df[Attr1].std()
    df[Attr2] = (df[Attr1] - mean) / std_dev


    mean = df[Attr1].mean()
    std_dev = df[Attr1].std()
    df[Attr2] = (df[Attr1] - mean) / std_dev

    max_abs = df[Attr1].abs().max()
    df[Attr2] = df[Attr1] / (10 ** len(str(int(max_abs))))



    if fl:
       my_data={
        "name":node[0].name,
        "result":"correlated",
        "p":p,
        "chi2":chi2,
        "dof":dof,
        "a1":Attr1,
        "a2":Attr2
       }

       return JsonResponse(my_data)
    
    my_data={
        "name":node[0].name,
        "result":"not correlated",
        "p":p,
        "chi":chi2,
        "dof":dof,
        "a1":Attr1,
        "a2":Attr2
       }

    return JsonResponse(my_data)



def assignment3(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    strr="info"
    
    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

   
    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety'] 

    if strr=="info":
       clf = DecisionTreeClassifier(criterion='entropy')
    elif strr=="gini":
       clf= DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=42)
    elif strr=="gain":
       clf = DecisionTreeClassifier(criterion='entropy', splitter='best')




    clf.fit(X, Y)
    decision_tree_text = export_text(clf, feature_names=X.columns.tolist())
    tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    tree_image_path = 'static/plot/image.png'
    os.makedirs(os.path.dirname(tree_image_path), exist_ok=True)
    plt.savefig(tree_image_path)
    my_data={"name":file_name,"text":decision_tree_text}
    return JsonResponse(my_data)
               




def assignment3_confuse_matrix(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='gini', splitter='best')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    misclassification_rate = 1 - accuracy
    sensitivity = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    response_data = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'misclassification_rate': misclassification_rate,
        'sensitivity': sensitivity,
        'precision': precision,
    }

    return JsonResponse(response_data, status=status.HTTP_200_OK)


def assignment4(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    clf.fit(X_train, y_train)


    rules = export_text(clf, feature_names=list(X.columns.tolist()))

    y_pred = clf.predict(X)
    accuracy = accuracy_score(Y, y_pred)

    coverage = len(y_pred) / len(Y) * 100

    rule_count = len(rules.split('\n'))

    my_data = {
        "name":file_name,
        'rules': rules,
        'accuracy': accuracy,
        'coverage': coverage,
        'toughness': rule_count,
    }
    return JsonResponse(my_data)


import csv
import io
import random
import math

def load_dataset(csv_file):
    dataset = []
    decoded_file = csv_file.read().decode('utf-8')
    io_string = io.StringIO(decoded_file)
    for row in csv.reader(io_string):
        if len(row) == 3:
            dataset.append([float(row[0]), float(row[1]), row[2]])
    return dataset

def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1) - 1):
        distance += (instance1[i] - instance2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(training_set, test_instance, k):
    distances = []
    for train_instance in training_set:
        dist = euclidean_distance(test_instance, train_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_class(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

def accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    if float(len(test_set))==0:
        return 0
    
    return (correct / float(len(test_set))) * 100.0


def assignment5(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    dataset = load_dataset(node[0].file)

    random.shuffle(dataset)
    split_index = int(len(dataset) * 0.8)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]

    k = 3
    predictions = []
    for test_instance in test_set:
            neighbors = get_neighbors(train_set, test_instance, k)
            result = predict_class(neighbors)
            predictions.append(result)

    accuracy_score = accuracy(test_set, predictions)

    return JsonResponse({'predictions': predictions, 'accuracy': accuracy_score}, status=status.HTTP_200_OK)




###################################################################################################

# views.py

# Chi-Square Value: 1922.9347363945576

# P-Value: 2.6830523867648017e-17

from scipy.stats import chi2_contingency,zscore,pearsonr
import tempfile
from django.shortcuts import render
import json
# Create your views here.
from rest_framework.parsers import FileUploadParser
import csv
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.views import View
import csv
import math
from django.http import JsonResponse
from django.views import View
import csv
from django.http import HttpResponse
import json
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv
import statistics
import numpy as np
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
import statistics
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,zscore,pearsonr
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import datasets
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.tree import export_text
from django.views.decorators.csrf import csrf_exempt
import logging
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import tempfile
import shutil
import math
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse, FileResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import os

import numpy as np
from django.http import JsonResponse
from scipy.stats import chi2_contingency
from scipy.stats import chi2

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import graphviz
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle


class RegressionClass(APIView):
    @method_decorator(csrf_exempt)
    def get(self, request, *args, **kwargs):
        if request.method == 'GET':
            try:
                
                node=CSVFile.objects.all()
                print(node[0].name)

                if len(node)==0 :
                    return HttpResponse("No csv file in database !!")

                print("boundary 1")

                algo="KNN"

                data = pd.read_csv(node[0].file)

                df = pd.DataFrame(data)
                df = shuffle(df, random_state=42)

                target_class = df.columns[-1]

                object_cols = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype=='object' and col != target_class]
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != target_class]

                X = df[numeric_cols+object_cols]
                y = df[target_class]

                # print(X.head())
                # print(y.head())
                

                X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
                ordinal_encoder = OrdinalEncoder()

                if target_class in object_cols :
                    object_cols = [col for col in object_cols if col != target_class]
                    y_train[target_class] = OrdinalEncoder.fit_transform(y_train[target_class])
                    y_test[target_class] = OrdinalEncoder.fit_transform(y_test[target_class])

                X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
                X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])


                if algo == "Linear" : 
                    cm = self.logistic_regression(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "Naive" : 
                    cm = self.naive_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "KNN" : 
                    cm = self.knn_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "ANN" : 
                    cm = self.ann_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                
                return JsonResponse({"accuracy": "accuracy"})
            except Exception as e :
                print(e)
                return JsonResponse({"error => ": str(e)}, status=status.HTTP_200_OK, safe=False)

    def preprocess(self, df):
        numerical_columns = df.select_dtypes(include=[int, float])

        # Select columns with a number of unique values less than 4
        unique_threshold = 4
        selected_columns = []
        for column in df.columns:
            if len(df[column].unique()) < unique_threshold and df[column].dtype == 'object':
                selected_columns.append(column)

        # Combine the two sets of selected columns (numerical and unique value threshold)
        final_selected_columns = list(set(numerical_columns.columns).union(selected_columns))

        # Create a new DataFrame with only the selected columns
        filtered_df = df[final_selected_columns]

        from sklearn.preprocessing import LabelEncoder

        # Assuming 'filtered_df' is your DataFrame with object-type columns to be encoded
        encoder = LabelEncoder()

        for column in filtered_df.columns:
            if filtered_df[column].dtype == 'object':
                filtered_df[column] = encoder.fit_transform(filtered_df[column])
        
        return filtered_df


    def logistic_regression(self, X, y, X_train, X_test, y_train, y_test):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # Load your dataset (e.g., Iris or Breast Cancer)
        # X, y, X_train, X_test, y_train, y_test = load_data()
        # Split the data into training and testing sets
        
        # Create and train the regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)

        print(y_pred.shape)
        print(y_test.shape)
        cm = confusion_matrix(y_test, y_pred).tolist()
        print(cm)

        
        return cm

    def naive_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score

        

        # Create and train the Naïve Bayes classifier
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = nb_classifier.predict(X_test)

        # Calculate accuracy
        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm

    def knn_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        

        # Create and train the k-NN classifier with different values of k
        k_values = [1, 3, 5, 7]
        cm = []
        accuracy_scores = []


        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = knn_classifier.predict(X_test)
            
            # Calculate accuracy
            metrix = confusion_matrix(y_test, y_pred).tolist()
            cm.append({'confusion_matrix': metrix})


            print(metrix)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # Plot the error graph
        plt.figure(figsize=(8, 6))  # Adjust figure size as needed
        plt.plot(k_values, accuracy_scores)
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        # plt.show()

        plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\KNNplot.png")
    
        return cm

    def ann_classifier(self, X, y, X_train, X_test, y_train, y_test):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_iris, load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier

        

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the ANN classifier
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)

        # Plot the error graph (iteration vs error)
        plt.figure(figsize=(8, 6))
        plt.plot(mlp.loss_curve_)
        plt.title('Error Graph (Iteration vs Error)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        # plt.show()

        plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\ANNplot.png")

        # Evaluate the classifier
        y_pred = mlp.predict(X_test)

        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm



import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

#####
def plot_dendrogram(Z):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\AGNES.png")


    # Save dendrogram plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Convert the plot to base64 for embedding in the API response
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # Close the plot to release resources
    plt.close()

    return img_base64


def agnes_clustering(data):
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(data)

    # Generate dendrogram
    linked = linkage(data, 'ward')
    dendrogram_plot = plot_dendrogram(linked)

    return labels, dendrogram_plot


######

def plot_dendrogram_diana(data, labels):
    plt.figure(figsize=(10, 7))
    linkage_matrix = np.zeros((len(data) - 1, 4))

    # Custom linkage matrix for DIANA
    for i in range(1, len(data)):
        parent = np.unique(labels[:i])[-1]
        children = np.unique(labels[i:])
        linkage_matrix[i - 1, 0] = parent
        linkage_matrix[i - 1, 1] = children[0]
        linkage_matrix[i - 1, 2] = i
        linkage_matrix[i - 1, 3] = len(np.where(labels == parent)[0]) + len(np.where(labels == children[0])[0])

    dendrogram(linkage_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram (DIANA)')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    # Save dendrogram plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Convert the plot to base64 for embedding in the API response
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # Close the plot to release resources
    plt.close()

    return img_base64


def perform_diana_clustering(data, n_clusters):
    def split_cluster(cluster_data):
        # Find the feature with maximum variance
        max_variance_feature = np.argmax(np.var(cluster_data, axis=0))

        # Sort data based on the chosen feature
        sorted_indices = np.argsort(cluster_data[:, max_variance_feature])

        # Split the cluster into two
        split_index = len(sorted_indices) // 2
        cluster1_indices = sorted_indices[:split_index]
        cluster2_indices = sorted_indices[split_index:]

        return cluster1_indices, cluster2_indices

    def recursive_diana(cluster_data, cluster_labels, remaining_clusters):
        if remaining_clusters == 1:
            return

        # Find the cluster with the highest variance
        max_variance_cluster = np.argmax([np.var(cluster_data[cluster_labels == cluster], axis=0).sum()
                                          for cluster in np.unique(cluster_labels)])

        # Split the cluster into two
        split_indices1, split_indices2 = split_cluster(cluster_data[cluster_labels == max_variance_cluster])

        # Update cluster labels
        cluster_labels[cluster_labels == max_variance_cluster] = max(cluster_labels) + 1

        # Recursively split the clusters
        recursive_diana(cluster_data[split_indices1], cluster_labels[split_indices1], remaining_clusters - 1)
        recursive_diana(cluster_data[split_indices2], cluster_labels[split_indices2], remaining_clusters - 1)

    # Initial clustering
    initial_labels = np.zeros(len(data))
    recursive_diana(data, initial_labels, n_clusters)

    # Generate dendrogram
    dendrogram_plot = plot_dendrogram_diana(data, initial_labels)

    return initial_labels, dendrogram_plot

@api_view(['GET'])
def hierarchical_clustering(request, method, format=None):
    # Sample data (replace this with your dataset)

    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)
    # data = np.array([
    #     [5.1, 3.5, 1.4, 0.2],
    #     [4.9, 3.0, 1.4, 0.2],
    #     [4.7, 3.2, 1.3, 0.2],
    #     [7.0, 3.2, 4.7, 1.4],
    #     [6.4, 3.2, 4.5, 1.5],
    #     [6.9, 3.1, 4.9, 1.5],
    #     [6.3, 3.3, 6.0, 2.5],
    #     [5.8, 2.7, 5.1, 1.9],
    #     [7.1, 3.0, 5.9, 2.1]
    # ])

    if method == 'agnes':
        labels, dendrogram_plot = agnes_clustering(data)
    elif method == 'diana':
        labels, dendrogram_plot = perform_diana_clustering(data,3)
    else:
        return Response({'error': 'Invalid clustering method'}, status=status.HTTP_400_BAD_REQUEST)

    return Response({'labels': labels.tolist(), 'dendrogram': dendrogram_plot,"name":node[0].name}, status=status.HTTP_200_OK)





def kmeans_clustering_algorithm(data, n_clusters, max_iters=100):
    num_points, num_features = data.shape

    # Initialize centroids randomly
    centroids = data[np.random.choice(num_points, n_clusters, replace=False)]

    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=-1), axis=-1)

        # Update centroids
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def plot_kmeans_clusters(data, labels, centroids):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue']

    # Plot data points
    for k in range(len(centroids)):
        cluster_points = data[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k + 1}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroids')

    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\KMEANS.png")
    return
    


@api_view(['GET'])
def kmeans_clustering(request, format=None):
    # Sample data (replace this with your dataset)
    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)
    

    n_clusters = 3

    labels, centroids = kmeans_clustering_algorithm(data, n_clusters)

    # Plot the results
    plot_kmeans_clusters(data, labels, centroids)

    return Response({'labels': labels.tolist(), 'centroids': centroids.tolist(),"name":node[0].name,}, status=status.HTTP_200_OK)










import numpy as np
import matplotlib.pyplot as plt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

def assign_points_to_medoids(data, medoids):
    distances = np.linalg.norm(data[:, np.newaxis, :] - medoids, axis=-1)
    labels = np.argmin(distances, axis=-1)
    return labels

def calculate_total_cost(data, labels, medoids):
    total_cost = 0
    for i, medoid in enumerate(medoids):
        cluster_points = data[labels == i]
        cluster_cost = np.linalg.norm(cluster_points - medoid, axis=-1).sum()
        total_cost += cluster_cost
    return total_cost

def plot_kmedoids_clusters(data, labels, medoids):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue']

    # Plot data points
    for k in range(len(medoids)):
        cluster_points = data[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k + 1}')

    # Plot medoids
    medoids = np.array(medoids)
    plt.scatter(medoids[:, 0], medoids[:, 1], marker='X', s=200, c='black', label='Medoids')

    plt.title('k-Medoids Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\KMEDOID.png")
    return


def kmedoids_clustering_algorithm(data, n_clusters, max_iters=100):
    num_points, num_features = data.shape

    # Initialize medoids randomly
    medoids = data[np.random.choice(num_points, n_clusters, replace=False)]
    labels = assign_points_to_medoids(data, medoids)

    for _ in range(max_iters):
        # Find the cost of the current clustering
        current_cost = calculate_total_cost(data, labels, medoids)

        # Randomly select a non-medoid point
        non_medoid_indices = np.setdiff1d(np.arange(num_points), medoids)
        random_non_medoid = np.random.choice(non_medoid_indices)

        for i, medoid in enumerate(medoids):
            # Swap the medoid with the non-medoid point
            medoids[i] = random_non_medoid

            # Recalculate labels and cost
            new_labels = assign_points_to_medoids(data, medoids)
            new_cost = calculate_total_cost(data, new_labels, medoids)

            # If the new clustering has a lower cost, accept the swap
            if new_cost < current_cost:
                labels = new_labels
                current_cost = new_cost
            else:
                # Revert the medoid swap
                medoids[i] = medoid

    return medoids, labels





@api_view(['GET'])
def kmedoids_clustering(request, format=None):
    # Sample data (replace this with your dataset)
    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)
    # data = np.array([
    #     [5.1, 3.5, 1.4, 0.2],
    #     [4.9, 3.0, 1.4, 0.2],
    #     [4.7, 3.2, 1.3, 0.2],
    #     [7.0, 3.2, 4.7, 1.4],
    #     [6.4, 3.2, 4.5, 1.5],
    #     [6.9, 3.1, 4.9, 1.5],
    #     [6.3, 3.3, 6.0, 2.5],
    #     [5.8, 2.7, 5.1, 1.9],
    #     [7.1, 3.0, 5.9, 2.1]
    # ])

    n_clusters = 3

    medoids, labels = kmedoids_clustering_algorithm(data, n_clusters)

    # Plot the results
    plot_kmedoids_clusters(data, labels, medoids)

    return Response({'medoids': medoids.tolist(), 'labels': labels.tolist(),"name":node[0].name,}, status=status.HTTP_200_OK)





def merge_clusters(clusters, cluster_centers, branching_factor):
    # Calculate pairwise distances between cluster centers
    distances = cdist(cluster_centers, cluster_centers)
    np.fill_diagonal(distances, np.inf)

    # Find the pair of clusters with the smallest distance
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    cluster1, cluster2 = min_indices

    # Merge the two clusters
    clusters[cluster1] += clusters[cluster2]
    cluster_centers[cluster1] = np.mean(clusters[cluster1], axis=0)

    # Remove the merged cluster
    del clusters[cluster2]
    del cluster_centers[cluster2]

    return clusters, cluster_centers

def plot_birch_clusters(data, clusters, cluster_centers):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue']

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i + 1}')

    cluster_centers = np.array(cluster_centers)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, c='black', label='Cluster Centers')

    plt.title('BIRCH Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\BIRCH.png")



def birch_clustering_algorithm(data, threshold, branching_factor):
    num_points, num_features = data.shape

    clusters = []
    cluster_centers = []  # Initialize as a list

    # Initialize the first cluster
    clusters.append([data[0]])
    cluster_centers.append(data[0])

    for i in range(1, num_points):
        point = data[i]

        # Find the nearest cluster center
        distances = np.linalg.norm(np.array(cluster_centers) - point, axis=1)
        nearest_cluster = np.argmin(distances)

        # Check if the point is within the threshold distance of the nearest cluster
        if distances[nearest_cluster] <= threshold:
            clusters[nearest_cluster].append(point)

            # Update the cluster center
            cluster_centers[nearest_cluster] = np.mean(clusters[nearest_cluster], axis=0)
        else:
            # If not, create a new cluster
            clusters.append([point])
            cluster_centers.append(point)

            # Adjust the number of clusters if the branching factor is exceeded
            if len(clusters) > branching_factor:
                clusters, cluster_centers = merge_clusters(clusters, cluster_centers, branching_factor)
    return clusters, np.array(cluster_centers) 





from scipy.spatial.distance import cdist


@api_view(['GET'])
def birch_clustering(request, format=None):
    # Sample data (replace this with your dataset)
    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)

    threshold = 1.0
    branching_factor = 3

    clusters, cluster_centers = birch_clustering_algorithm(data, threshold, branching_factor)

    # Plot the results
    plot_birch_clusters(data, clusters, cluster_centers)

    return Response({'clusters': clusters, 'cluster_centers': cluster_centers,"name":node[0].name,}, status=status.HTTP_200_OK)







def find_neighbors(data, point_index, eps):
    distances = np.linalg.norm(data - data[point_index], axis=1)
    return np.where(distances <= eps)[0]

def expand_cluster(data, labels, point_index, neighbors, cluster_id, eps, min_samples):
    labels[point_index] = cluster_id

    i = 0
    while i < len(neighbors):
        current_point = neighbors[i]

        if labels[current_point] == -1:
            labels[current_point] = cluster_id
        elif labels[current_point] == 0:
            labels[current_point] = cluster_id

            new_neighbors = find_neighbors(data, current_point, eps)

            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate([neighbors, new_neighbors])
        
        i += 1  # Move the increment inside the loop

def group_clusters(labels):
    unique_labels = np.unique(labels)
    clusters = {}

    for label in unique_labels:
        if label == -1:
            continue

        cluster_points = np.where(labels == label)[0]
        clusters[label] = cluster_points.tolist()

    return clusters

def plot_dbscan_clusters(data, clusters):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']

    for i, cluster_points in enumerate(clusters.values()):
        cluster_points = np.array([data[index] for index in cluster_points])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i + 1}')

    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\DBSCAN.png")



def dbscan_clustering_algorithm(data, eps, min_samples):
    num_points, num_features = data.shape

    labels = np.zeros(num_points, dtype=int)
    cluster_id = 0

    for i in range(num_points):
        if labels[i] != 0:
            continue

        neighbors = find_neighbors(data, i, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels, group_clusters(labels)



@api_view(['GET'])
def dbscan_clustering(request, format=None):
    # Sample data (replace this with your dataset)

    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)

    eps = 0.5
    min_samples = 3

    labels, clusters = dbscan_clustering_algorithm(data, eps, min_samples)

    # Plot the results
    plot_dbscan_clusters(data, clusters)

    return Response({'labels': labels.tolist(), 'clusters': clusters,"name":node[0].name,}, status=status.HTTP_200_OK)


import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch, DBSCAN

@api_view(['GET'])
def clustering_evaluation(request, format=None):
    # Load Iris dataset from CSV

    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)

    # Extract ground truth labels (assuming the last column is the ground truth labels)
    ground_truth_labels = iris_data.iloc[:, -1].values

    # Hierarchical Clustering (Agglomerative)
    agnes = AgglomerativeClustering(n_clusters=3)
    agnes_labels = agnes.fit_predict(data)
    agnes_silhouette = silhouette_score(data, agnes_labels)
    agnes_rand_index = adjusted_rand_score(ground_truth_labels, agnes_labels)

    # K-Means
    kmeans = KMeans(n_clusters=3)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_silhouette = silhouette_score(data, kmeans_labels)
    kmeans_rand_index = adjusted_rand_score(ground_truth_labels, kmeans_labels)

    # Birch
    birch = Birch(n_clusters=3)
    birch_labels = birch.fit_predict(data)
    birch_silhouette = silhouette_score(data, birch_labels)
    birch_rand_index = adjusted_rand_score(ground_truth_labels, birch_labels)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    dbscan_labels = dbscan.fit_predict(data)
    dbscan_silhouette = silhouette_score(data, dbscan_labels)
    dbscan_rand_index = adjusted_rand_score(ground_truth_labels, dbscan_labels)

    # Create a dictionary with the results
    results = {
        "name":node[0].name,
        "Agg": {"Silhouette Score": agnes_silhouette, "Adjusted": agnes_rand_index},
        "KMeans": {"Silhouette Score": kmeans_silhouette, "Adjusted": kmeans_rand_index},
        "Birch": {"Silhouette Score": birch_silhouette, "Adjusted": birch_rand_index},
        "DBSCAN": {"Silhouette Score": dbscan_silhouette, "Adjusted": dbscan_rand_index}
    }


    return Response(results, status=status.HTTP_200_OK)










#################################################################################################




import pandas as pd
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from mlxtend.frequent_patterns import apriori, association_rules


# Import necessary libraries
import pandas as pd
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

def convert_to_json_serializable(value):
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32)):
        return float(value)
    else:
        return value

from graphviz import Digraph


def custom_association_rules(frequent_itemsets, min_threshold=0.7):
    rules = []

    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                antecedent = frozenset(itemset[:i])
                consequent = frozenset(itemset[i:])
                support_antecedent = frequent_itemsets[antecedent]
                support_itemset = frequent_itemsets[itemset]
                confidence = support_itemset / support_antecedent

                if confidence >= min_threshold:
                    lift = confidence / (frequent_itemsets[consequent] / frequent_itemsets[itemset])
                    leverage = support_itemset - (support_antecedent * frequent_itemsets[consequent])
                    conviction = (1 - frequent_itemsets[consequent]) / (1 - confidence)

                    rule = {
                        "antecedents": list(antecedent),
                        "consequents": list(consequent),
                        "antecedent support": support_antecedent,
                        "consequent support": frequent_itemsets[consequent],
                        "support": support_itemset,
                        "confidence": confidence,
                        "lift": lift,
                        "leverage": leverage,
                        "conviction": conviction,
                    }
                    rules.append(rule)

    return rules



def frozenset_to_list(frozenset_obj):
    return list(frozenset_obj)
    
@api_view(['GET'])
def generate_rules(request, format=None):
    # Load the dataset
    dataset_path = 'static/UsElections.csv'
    df = pd.read_csv(dataset_path)

    # Replace question marks with 0

    # Exclude the target column (assuming the first column is the target)
    target_column = df.columns[0]
    df_features = df.drop(target_column, axis=1)

    frequent_itemsets = apriori(df_features, min_support=0.1, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # Convert rules to a list of dictionaries for serialization
    print(rules)

   

    rules['antecedents'] = rules['antecedents'].apply(frozenset_to_list)
    rules['consequents'] = rules['consequents'].apply(frozenset_to_list)



    rules_list = rules.to_dict(orient='records')


      



    return JsonResponse({'association_rules': rules_list})



# association_rules_app/views.py
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from mlxtend.frequent_patterns import apriori, association_rules

@api_view(['GET'])
def generate_interesting_rules(request, format=None):
    # Load the dataset
    dataset_path = 'static/UsElections.csv'
    df = pd.read_csv(dataset_path)

    target_column = df.columns[0]
    df_features = df.drop(target_column, axis=1)

    # Preprocess the dataset if needed
    # (e.g., handle missing values, convert categorical variables, etc.)

    # Experiment with different values of support, confidence, and maximum rule length
    min_support = 0.05  # Adjust as needed
    min_confidence = 0.6  # Adjust as needed

    # Apriori algorithm
    frequent_itemsets = apriori(df_features, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
     
    rules['antecedents'] = rules['antecedents'].apply(frozenset_to_list)
    rules['consequents'] = rules['consequents'].apply(frozenset_to_list)

    # Calculate additional metrics
    rules['lift'] = rules['lift']
    # rules['chi_squared'] = rules['chi2']
    rules['all_confidence'] = rules['antecedent support'] / rules['consequent support']
    rules['max_confidence'] = rules[['antecedent support', 'consequent support']].max(axis=1)
    rules['kulczynski'] = 0.5 * (rules['antecedent support'] / rules['consequent support'] + rules['consequent support'] / rules['antecedent support'])
    
    print(rules['all_confidence'])


    
    # Select interesting rules based on the metrics
    interesting_rules = rules[(rules['lift'] > 1)  & (rules['all_confidence'] > 0) & (rules['max_confidence'] > 0) & (rules['kulczynski'] > 0)]

    # Convert rules to a list of dictionaries for serialization
    interesting_rules_list = interesting_rules.to_dict(orient='records')

    return JsonResponse({'interesting_rules': interesting_rules_list})




import requests
from bs4 import BeautifulSoup



def web_crawler(seed_url, method='bfs',max_depth=3):
    # Implement DFS or BFS crawler logic here
    # Return a list of crawled links
    # For simplicity, I'm using a basic example of BFS crawler
    visited = set()
    queue = [(seed_url,0)]
    links = []

    while queue:
        current_url, depth = queue.pop(0)


        if current_url not in visited and depth<=max_depth:
            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                links_on_page = [a['href'] for a in soup.find_all('a', href=True)]
                links.extend(links_on_page)
                visited.add(current_url)
                queue.extend([(link, depth + 1) for link in links_on_page])
            except Exception as e:
                print(f"Error crawling {current_url}: {e}")

    return links

import json

@csrf_exempt
def crawl(request):

    
    try:
         
        seed_url = request.POST.get('seed_url','')  # Adjust this based on how the seed_url is passed in the request
        print(seed_url)
        
        if seed_url:
            links = web_crawler(seed_url)
            # print(links)
            return JsonResponse({'links': links})
        else:
            return JsonResponse({"msg": "Missing 'seed_url' parameter in the request"}, status=400)


    except Exception as e:
        print(e)
        return JsonResponse({"msg":"Error occurred "})


import networkx as nx


def calculate_pagerank(request):
    try:
        # Read CSV file and create a directed graph
        G = nx.DiGraph()
        with open("C:/Users/Saurabh/Desktop/DM assignments/dm_assignments/backend/dm_pro/NodeEdges.csv",mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                from_node = row['FromNodeId']
                to_node = row['ToNodeId']
                G.add_edge(from_node, to_node)

        # Calculate PageRank
        pagerank_scores = nx.pagerank(G)

        # Get the 10 pages with the highest rank
        top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Tabulate the results containing the adjacency matrix and rank of pages
        adjacency_matrix = nx.adjacency_matrix(G).todense().tolist()
        rank_table = [{'Page': page, 'Rank': rank} for page, rank in top_pages]

        return JsonResponse({'rank_table': rank_table})
        return JsonResponse({'msg':'Request processed.....'})
    except Exception as e:
        print(e)
        return JsonResponse({"msg":"Error occurred "})



@csrf_exempt
def calculate_hits(request):
    try:
        # Path to the CSV file
        csv_file_path = 'C:/Users/Saurabh/Desktop/DM assignments/dm_assignments/backend/dm_pro/NodeEdges.csv'

        # Read edges from the CSV file
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row
            edges = [(int(row[0]), int(row[1])) for row in csv_reader]

        # Create a directed graph using NetworkX
        G = nx.DiGraph(edges)

        # Run the HITS algorithm
        hubs, authorities = nx.hits(G)

        # Get the top 10 authorities and hubs
        top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:10]
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:10]

        # Convert the graph to an adjacency matrix
        adj_matrix = nx.to_numpy_array(G)

        # Tabulate the results
        results = {
            # 'adj_matrix': adj_matrix.tolist(),
            'authority_rank': top_authorities,
            'hub_rank': top_hubs,
        }
        return JsonResponse(results)
    except Exception as e:
        print(e)
        return JsonResponse({"msg":"Error occurred "})


from .crawler import crawl_dfs, crawl_bfs

class CrawlAPIView(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        seed_url = data.get('seed_url', '')
        max_depth = data.get('max_depth', 2)
        method = data.get('method', 'dfs')

        if method == 'dfs':
            crawled_urls = crawl_dfs(seed_url, max_depth)
        elif method == 'bfs':
            crawled_urls = crawl_bfs(seed_url, max_depth)

        return JsonResponse({'crawled_urls': crawled_urls})








import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json

def frozenset_to_list(frozenset_item):
    return list(frozenset_item)

@method_decorator(csrf_exempt, name='dispatch')
class GenerateRulesView(View):
    def post(self, request, *args, **kwargs):
        try:
            # Load the dataset
            dataset_path = 'static/UsElections.csv'
            df = pd.read_csv(dataset_path)

            # Replace question marks with 0

            # Exclude the target column (assuming the first column is the target)
            target_column = df.columns[0]
            df_features = df.drop(target_column, axis=1)

            # Extract parameters from POST data
            min_support = request.POST.get('min_support', 0.1)
            min_confidence = request.POST.get('min_confidence', 0.7)
            max_rule_length = request.POST.get('max_rule_length', None)

            frequent_itemsets = apriori(df_features, min_support=float(min_support), use_colnames=True)

            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=float(min_confidence))

            # Convert rules to a list of dictionaries for serialization
            rules['antecedents'] = rules['antecedents'].apply(frozenset_to_list)
            rules['consequents'] = rules['consequents'].apply(frozenset_to_list)

            rules_list = rules.to_dict(orient='records')
            return JsonResponse({'association_rules': rules_list})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)



