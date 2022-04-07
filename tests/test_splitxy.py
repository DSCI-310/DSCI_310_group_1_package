from grouponefunctions import grouponefunctions
import pytest
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
import warnings

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
location, _ = urllib.request.urlretrieve(URL)
compressed_file = zipfile.ZipFile(location)
csv_file = compressed_file.open('student-mat.csv')
df = pd.read_csv(csv_file,sep = ";")

train_df, test_df = train_test_split(df, test_size = 0.2, random_state=100)

#make training and testing split
desiredfeatures = ["studytime", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "goout","romantic","traveltime"]


class Test_splitxy:
    #tests made using pytest documentation https://docs.pytest.org/en/7.0.x/ 
    #and pytest warning documentation https://docs.pytest.org/en/6.2.x/warnings.html
    
    def test_desiredfeatures_list(self):
        X_train, y_train = grouponefunctions.split_xy(train_df, desiredfeatures, "G3")
        X_test, y_test = grouponefunctions.split_xy(test_df, desiredfeatures, "G3")
        
        X_train2 = train_df[desiredfeatures]
        X_test2 = test_df[desiredfeatures]
        
        print("testing whether passing in a list of desiredfeatures produces correct output" )
        assert_frame_equal(X_train, X_train2)
        assert_frame_equal(X_test, X_test2)
    
    def test_desiredfeatures_string(self):
        X_train2 = train_df["romantic"]
        X_test2 = test_df["G2"]
        
        print("testing whether passing in a list of desiredfeatures produces correct output" )
        X_train, _= grouponefunctions.split_xy(train_df, "romantic", "G3")
        X_test, _ = grouponefunctions.split_xy(test_df, "G2", "G3")
        assert_series_equal(X_train, X_train2)
        assert_series_equal(X_test, X_test2)
    
    def test_target_list(self):
        # making a new list of features
        samplefeatures = ["studytime", "Pstatus", "Medu"]
        y_train2 = train_df[samplefeatures]
        y_test2 = test_df[samplefeatures]
        
        print("testing whether passing in a single produces correct output")
        _, y_train= grouponefunctions.split_xy(train_df, "Mjob", samplefeatures)
        _, y_test = grouponefunctions.split_xy(test_df, "Mjob", samplefeatures)
        assert_frame_equal(y_train, y_train2)
        assert_frame_equal(y_test, y_test2)
    
    def test_target_string(self):
        y_train2 = train_df["G3"]
        y_test2 = test_df["G3"]
        
        print("testing whether passing in a single produces correct output")
        _, y_train = grouponefunctions.split_xy(train_df, desiredfeatures, "G3")
        _, y_test = grouponefunctions.split_xy(test_df, desiredfeatures, "G3")
        assert_series_equal(y_train, y_train2)
        assert_series_equal(y_test, y_test2)
    
    def test_faultyvariable_in_list(self):
        print("make sure that we properly get an error when inputting desired/target feature we don't have within a list")
        with pytest.raises(KeyError):
            X_train, y_train = grouponefunctions.split_xy(train_df, ["G2", "asodfiajsdofi"], "G3")
            X_train, y_train = grouponefunctions.split_xy(train_df, "G3", ["G2", "asodfiajsdofi"])
    
    def test_faulty_single_variable(self):
        print("make sure that we properly get an error when inputting single desired/target feature we don't have")
        with pytest.raises(KeyError):
            X_train, y_train = grouponefunctions.split_xy(train_df, desiredfeatures, "There_will_be_nothing_here")
            X_train, y_train = grouponefunctions.splitxy(train_df, "There_will_be_nothing_here", desiredfeatures)
    
    def test_incorrect_inputtype(self):
        print("make sure that each of our inputs correctly throws a type error")
        print("throw type error when first input is not a df")
        print("throw type error when second or third input is not a list/str")
        with pytest.raises(TypeError):
            X_train, y_train = grouponefunctions.split_xy(3, desiredfeatures, "G3")
            X_train, y_train = grouponefunctions.split_xy(train_df, np.array([1, 2, 3, 4]), "G3")
            X_train, y_train = grouponefunctions.split_xy(train_df, "G3", np.array([1, 2, 3, 4]))
    
    def test_common_features(self):
        print("make sure we dont have warning if we dont have a repeat feature")
        X_train2, y_train= grouponefunctions.split_xy(train_df, "G3", "G2")
        with pytest.warns(UserWarning):
            print("make sure we do have warning if we have a repeat feature")
            samplefeatures = ["studytime", "Pstatus", "G3"]
            X_train, y_train= grouponefunctions.split_xy(train_df, "G3", samplefeatures)
            X_train2, y_train= grouponefunctions.split_xy(train_df, "G3", "G3")
            X_test, y_test = grouponefunctions.split_xy(test_df, samplefeatures, "G3")
    
    def test_empty(self):
        print("empty dataframe")
        dfObj = pd.DataFrame(columns=['User_ID', 'UserName', 'Action'])
        X_train, y_train = grouponefunctions.split_xy(dfObj, 'User_ID', "Action")