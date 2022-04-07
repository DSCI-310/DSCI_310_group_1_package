from grouponefunctions import grouponefunctions
import pytest
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
import matplotlib.pyplot as plt
import warnings


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
location, _ = urllib.request.urlretrieve(URL)
compressed_file = zipfile.ZipFile(location)
csv_file = compressed_file.open('student-mat.csv')
df = pd.read_csv(csv_file,sep = ";")

train_df, test_df = train_test_split(df, test_size = 0.2, random_state=100)
#make training and testing split
desiredfeatures = ["studytime", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "goout","romantic","traveltime"]
X_train, y_train = grouponefunctions.split_xy(train_df, desiredfeatures, "G3")
X_test, y_test = grouponefunctions.split_xy(test_df, desiredfeatures, "G3")

class Test_Square_Plot:
    def check_axs_equal(self, ax1, ax2):
        retval = True
        if ax1.shape != ax2.shape:
            return False
        for i in range(ax1.shape[0]):
            for j in range(ax1.shape[1]):
                if ax1[i,j].title.get_text() != ax2[i,j].title.get_text() and ax1[i,j].get_gridspec() != ax2[i,j].get_gridspec():
                    return False
        return True
        
    def test_good_5items_incomplete(self):
        
        fig, axs = plt.subplots(2, 3, figsize=(10,10))
        axs[0, 0].scatter(X_train["studytime"], y_train)
        axs[0, 0].set_title('study time vs grade')
        axs[0, 1].scatter(X_train["Medu"], y_train)
        axs[0, 1].set_title('Mother education vs grade')
        axs[0, 2].scatter(X_train["Fedu"], y_train)
        axs[0, 2].set_title('Father education vs grade')
        axs[1, 0].scatter(X_train["goout"], y_train)
        axs[1, 0].set_title('time spent with friends vs grade')
        axs[1, 1].scatter(X_train["traveltime"], y_train)
        axs[1, 1].set_title('travel time vs grade')
        txt = "Figure 2 A series of plots examining the numeric features compared to predicted grade"
        plt.figtext(0.5, 0.05, txt, wrap=True, horizontalalignment='center', fontsize=12)

        test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["studytime", "Medu", "Fedu", "goout", "traveltime"], 
                                    ["study time vs grade", "Mother education vs grade", "Father education vs grade", 
                                     "time spent with friends vs grade", "travel time vs grade"], txt)
        
        
        
        print("testing whether a set of 5 plots plots correctly in size and title (plots have some room for placement case)" )
        assert self.check_axs_equal(axs, test_axs)
    
    def test_good_4items_complete(self):
        
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        axs[0, 0].hist(X_train["Pstatus"])
        axs[0, 0].set_title('P status vs grade')
        axs[0, 1].hist(X_train["Mjob"])
        axs[0, 1].set_title('Mother job vs grade')
        axs[1, 0].hist(X_train["Fjob"])
        axs[1, 0].set_title('Father Job vs grade')
        axs[1, 1].hist(X_train["romantic"])
        axs[1, 1].set_title('Relationship status vs grade')
        txt = "Figure 3 A series of histograms examining the distribution of categorical features"
        plt.figtext(0.5, 0.05, txt, wrap=True, horizontalalignment='center', fontsize=12)

        #plt.savefig('../CorrectA.png')

        test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], txt)
        
        
        print("testing whether a set of 4 plots plots correctly in size and title (every plot in perfect square)" )
        assert self.check_axs_equal(axs, test_axs)
        

    def test_bad_not_dataframes_1(self):
        
        print("testing wrong input on on initial dataframes (first)" )
        
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, "hello", ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], "Sample")
            assert str(exc_info.value) == 'The first two arguments are not dataframes of equal length'
            
    def test_bad_not_dataframes_2(self):
        
        print("testing wrong input on on initial dataframes (second)" )
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data("hello", y_train, ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], "Sample")
            assert str(exc_info.value) == 'The first two arguments are not dataframes of equal length'
            
    def test_bad_dataframes_size(self):
        
        print("testing that dataframes dont work when not same length" )
        
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, pd.DataFrame(), ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], "Sample")
            assert str(exc_info.value) == 'The first two arguments are not dataframes of equal length'
            
            
    def test_bad_not_desired_not_list(self):
        
        print("testing that desired inputs is appropriate list" )
        
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, [], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], "Sample")
            assert str(exc_info.value) == 'desiredFeatures is not a list of strings length at least 1'
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, [1,2,3], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], "Sample")
            assert str(exc_info.value) == 'desiredFeatures is not a list of strings length at least 1'
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["Pstatus"], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], "Sample")
            assert str(exc_info.value) == 'desiredFeatures is not a list of strings length at least 1'
            
    def test_bad_titles(self):
        
        print("testing that desired titles is appropriate list" )
        
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    [], "Sample")
            assert str(exc_info.value) == 'titles is not a list of strings of length equal to desiredFeatures'
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    [1,2,3], "Sample")
            assert str(exc_info.value) == 'titles is not a list of strings of length equal to desiredFeatures'
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    ["alpha"], "Sample")
            assert str(exc_info.value) == 'titles is not a list of strings of length equal to desiredFeatures'
            
    def test_bad_not_txt_not_string(self):
        
        print("testing that last argument crashes on incorrect input" )
        
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["Pstatus", "Mjob", "Fjob", "romantic"], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], 123)
            assert str(exc_info.value) == 'The last argument is not a string'
            
    def test_bad_feature_dne(self):
        
        print("testing for invalid column names as desiredFeature" )
        
        with pytest.raises(TypeError) as e_info:
            test_axs, _ = grouponefunctions.plot_square_data(X_train, y_train, ["china", "Mjob", "Fjob", "romantic"], 
                                    ["P status vs grade", "Mother job vs grade", "Father Job vs grade", "Relationship status vs grade"], "Sample")
            assert str(exc_info.value) == 'desiredFeature is not in dependent dataframe'