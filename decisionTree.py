import numpy as np
import sklearn
import sklearn.model_selection
import sys
from os import path
import subprocess
import graphviz
from imblearn.combine import SMOTEENN


if(len(sys.argv) >= 3):
    numberOfLines = int(sys.argv[2])
    matrixFile = sys.argv[1]
elif(len(sys.argv) >= 2 and (sys.argv[1] != "--help" or sys.argv[1] != "-h" or sys.argv[1] != "-help")):
    numberOfLines = -1
    matrixFile = sys.argv[1]
else:
    print("usage : python3 decisionTree.py matrixFile [numberOfLines]\n\t-matrixFile is the file containing the matrix of clusters (tsv format)\n\t-If given, numberOfLines is a positive number, representing the number of lines that the script will read (defines the shape of matrix)")
    exit(0)


def partition(numberOfLines, matrixFile):
    """
    param : numberOfLines is an integer and represents, if that number is positive, the number of lines of the matrix file that the script will.
    If that's not the case, the function will read all the lines of the matrix file
    Reads matrix file and returns a list with 3 objects :
    - The first (transposeMatrix) is the transpose of the matrix (which is an array). This matrix doesn't contain the Nox cluster.
    - The second (noxCluster) is the line of the Nox cluster.
    - The third (FeaturesNames) is an array of cluster names inside the first object.
    """

    if(numberOfLines > 0):
        print("Reading ", numberOfLines, " lines of the matrix file.")
    else :
        "Full Reading of the matrix file."

    # Opening file
    with open(matrixFile) as f:
        print("Reading file...")
        # read sthe first line to skip it (The first line contains the names of the organisms, not needed (yet) for the script.
        f.readline()
        ligne = f.readline().rstrip()
        matrix = []
        featuresNames = []
        while(ligne or len(matrix) < numberOfLines):
            if(len(matrix) > 0 and len(matrix) < numberOfLines and len(matrix) % (numberOfLines/8) == 0):
                print(len(matrix), "lines read...")

            t = ligne.split("\t")
            # The cluster n°26583 is the NOX cluster, we have to save the line outside the matrix (for testing with decision tree algorithm)
            if(t[0] == "26583"):
                print("Reading nox cluster line")
                noxCluster = t[1:]
            else:
                # Any line that's not the Nox cluster
                if(numberOfLines > 0):
                    if(len(matrix) < numberOfLines):
                        # if the matrix isn't full (len(matrix)!=numberOfLines) and the numberOfLines is positive
                        matrix.append(t[1:])
                        featuresNames.append("Cluster n°"+t[0])
                else:
                    # We're in the case where numberOfLines is negative, so we're reading all the lines of matrixFile
                    if(len(matrix) % 10000 == 0):
                        print(len(matrix), "readed lines")
                    matrix.append(t[1:])
                    featuresNames.append("Cluster n°"+t[0])
            # reading the next line
            ligne = f.readline().rstrip()

    transposeMatrix = np.array(matrix).transpose()
    return (transposeMatrix, noxCluster, featuresNames)


data = partition(numberOfLines, matrixFile)

# Decision Tree
''' Transforming the data '''
print("Transforming data for the decision tree...")
# We need an array of int in order to use it with the sklearn.tree
y = list(map(int, data[1]))
x = []
# We need an array of int in order to use it with the sklearn.tree
for tab in data[0]:
    x.append(list(map(int, tab)))


if(np.array(x).shape[0] != np.array(y).shape[0]):
    print(
        "The decision tree can't be created if x and y don't have the same length / shape[0]")
    exit(-1)
else:
    print("Shape x (Matrix) : ", np.array(x).shape)
    print("Shape y (Nox cluster, line matrix) : ", np.array(y).shape)

'''Creating the decision tree and spliting data for training and testing'''
print("Creating the decision tree and spliting data.")
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=100)

'''oversampling et undersampling combines'''
comb_samp = SMOTEENN()

'''Training data for the decision tree'''
print("Training the decision tree...")
x_comb, y_comb = comb_samp.fit_sample(x_train, y_train)

classifier_entropy = sklearn.tree.DecisionTreeClassifier(
    criterion="entropy", random_state=100)
classifier_entropy.fit(x_comb, y_comb)

y_pred = classifier_entropy.predict(x_test)

accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
# Printing the accuracy of the decision tree
print("Accuracy classification score of the decision tree :", str(accuracy*100)+"%.")


'''Creation of a graphic view of the decision tree with graphviz'''

dot_data = sklearn.tree.export_graphviz(classifier_entropy, out_file="DecisionTree.dot", class_names=[
                                        "Not Nox", "Nox"], feature_names=data[2], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
# The output file is in dot format so we need to transform it
subprocess.check_call(
    ["dot", "-T", "png", "DecisionTree.dot", "-o", "DecisionTree.png"])
if(path.exists("DecisionTree.dot") and path.exists("DecisionTree.png")):
    print("The script creats in the current directory the DecisionTree.png file which is the graphic view of the decision tree.")
    exit(0)
elif(path.exists("DecisionTree.dot")):
    print("The script creats the DecisionTree.dot file but hasn't transformed it into a png file, in order to transform it you have to execute the command: \ndot -T png DecisionTree.dot -o DecisionTree.png")
    exit(0)
