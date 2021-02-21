github repository:
https://github.com/jlewis200/CS7641_Project_1

run with:
./main.py

depends on:
matplotlib, sklearn, numpy, pandas, multiprocessing

main.py - This is the main entry point for the program.  It provides the best parameters for each classifier, and classifier-kernel/classifier-solver combination when comparisons are made.  It also contains the main plot-producing functions.  The plots are saved to *.png files directly.  Runtime is approximately 30 minutes on a 96 core Amazon Web Services c5a.24xlarge.

parameter_utils.py - This file contains the supporting code to cross validate the models with a variety of different parameters.  The main functions are get_params_[DATASET]().  This returns a dictionary of dictionaries, each of which contains the best parameters (among those tested) for a given model.  These can be passed to the classifier constructor directly by using the ** unpacking operator. Example use:
    params = get_params_se()
    classifier = SVC(**params['svm_params_linear']), *data)

dataset_utils.py - This file contains code to format, normalize, and split the data into train/test splits.  It uses a caching system to avoid loading the data from file and normalizing more than once.  The main function is get_normalized_data(file_name, train_ratio).  It returns a 4-tuple of numpy arrays containing training feature vectors, training classifications, testing feature vectors, and testing classifications.
