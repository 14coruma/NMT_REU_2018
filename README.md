# PDF Malware Detection Using Visualization and Image Processing Techniques
By Andrew Corum (Calvin College) and Donovan Jenkins (New Mexico Tech)

## Dependencies
* Python3
* OpenCV
* scikit-learn (sklearn)
* scikit-image (skimage)
* SciPy (scipy)
* multiprocessing

## Usage
```
./main.py <dirname>
```

**Note:** Within the directory, \<dirname\>, PDFs must be labeled as *CLEAN* or
*INFEC* (i.e. "CLEAN_\<filename\>.file" or "INFEC_\<filename\>.file"). They also must
end with the ".file" tag.

**Note:** It may take a while for the program to start on your first run, since python
needs to load all of the necessary libraries.

### Preprocessing
PDFs must first be converted into images. You can either load pre-created images or create new images.
```
Select an option:
    0) Load
    1) Create
```

PDFs can be visualized as a byte plot, Markov plot, or Hilbert plot.
```
Choose a visualization type
    0) Byte
    1) Markov
    2) Hilbert
```

### Feature extraction
You will be prompted for which features to extract from the visualized PDFs.  
The options are Oriented Rotated BRIEF (ORB), Scale Invariant Feature Transform (SIFT), 
Local Binary Patterns (LBP), Gabor filter and histogram, and Local Entropy.
```
Choose a feature selection algorithm:
    0) ORB
    1) SIFT
    2) LBP
    3) Gabor
    4) Entropy
```

### Machine learning
This program allows you to either perform cross validation on your data or train a 
model that will then be used to evaluate other PDFs.
```
Select an option:
    0) Cross validation
    1) Build and test model
```

You have a choice of which machine learning algorithm to use. 
```
Choose a ML algorithm:
    0) Support Vector Machine
    1) Random Forest
    2) Decision Tree Classifier
    3) KNN
```

If you chose to build and test a model, you will have to repeat the preprocessing and
feature extraction steps on your test data. If you chose cross validation, then you will 
see the accuracy, precision, recall, and f1 score of your model.
