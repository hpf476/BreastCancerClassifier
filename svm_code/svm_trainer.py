import os
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import sklearn
from skimage import io
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import glob

# train_file = '/home/ubuntu/workspace_ami/breakhis_data/train_val_test_60_12_28/non_shuffled/split1/200X_train.txt'
# val_file = '/home/ubuntu/workspace_ami/breakhis_data/train_val_test_60_12_28/non_shuffled/split1/200X_val.txt'
# test_file = '/home/ubuntu/workspace_ami/breakhis_data/train_val_test_60_12_28/non_shuffled/split1/200X_test.txt'
train_file = '40X_train.txt'
val_file = '40X_val.txt'
test_file = '40X_test.txt'
dir_path = '/home/pf1404/Documents/ai_coding_tasks/project/breakhis_data/'

def combine_file_paths(file1, file2):
    file_list = [file1,file2]
    print (file_list)
    with open("combined_image_path.txt", "w") as outfile:
        for fname in file_list:
            with open(fname) as infile:
                for line in infile:
                    line_write = os.path.join(dir_path, line)
                    outfile.write(line_write)

def read_class_list_mod(class_list):
    """
    Scan the image file and get the image paths and labels
    """
    with open(class_list) as f:
        lines = f.readlines()
        images = []
        labels = []
        for l in lines:
            items = l.strip().split()
            #z=os.path.basename(items[0])
            z=items[0]
            images.append(z)
            labels.append(int(items[1]))
    return images, labels

def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.

    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def)


def extract_features(image_paths, verbose=True):
    """
    extract_features computed the inception bottleneck feature for a list of images

    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))

    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name("import/pool_3:0")
        for i, image_path in enumerate(image_paths):
            if verbose:
                print('Processing %s...' % (image_path))

            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)

            image_data = io.imread(image_path) # read() returns content as string
            # image_data = gfile.FastGFile(image_path, 'rb').read() # read() returns content as string
            feature = sess.run(flattened_tensor, {
                "import/DecodeJpeg:0": image_data
            })
            features[i, :] = np.squeeze(feature)

    return features

def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)

    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]

    # request probability estimation
    svm = SVC(probability=True)

    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
            cv=10, n_jobs=4, verbose=3)

    clf.fit(X_train, y_train)

    if os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict=clf.predict(X_test)

    labels=sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

model_path = "/home/pf1404/Documents/ai_coding_tasks/project/BreastCancerClassifier/svm_code_pf/output_graph.pb"
image_paths = '/home/pf1404/Documents/ai_coding_tasks/project/BreastCancerClassifier/svm_code_pf/combined_image_path.txt'
model_output_path = "'/home/pf1404/Documents/ai_coding_tasks/project/BreastCancerClassifier/svm_code_pf/"
combine_file_paths(train_file, val_file)

images, labels = read_class_list_mod(image_paths)
create_graph(model_path)
features = extract_features(images)

train_svm_classifer(features, labels, model_output_path)
