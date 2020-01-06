# Replication source code for BiLSTM

This folder contains the necessary sources for running the baseline system.
The file bilstm_baseline.py contains the classifier, 
it uses the dataset from the [data folder](code/data/)
and the [tf_data.py](tools/tf_data.py) from the tools module found in the [tools folder](code/tools/) folder.

To successfully run the baseline it is necessary to add the [portuguese fasttext embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) to the [data folder](code/data) and rename it to *fasttextpt.vec*.

An example of an expected output can be found in the file [bilstm_baseline_output.txt](code/bilstm_baseline_output.txt).

```
You may require:

python                  3.6.7
gensim                  3.7.3   
gpt-2-simple            0.6    
h5py                    2.9.0  
Keras                   2.2.5   
Keras-Applications      1.0.8   
Keras-Preprocessing     1.1.0   
lxml                    4.3.3   
mathutils               2.81.2  
matplotlib              3.1.1   
nltk                    3.4.1   
numpy                   1.16.4  
pandas                  0.24.2  
pip                     19.2.3  
protobuf                3.8.0   
regex                   2019.6.8
tensorboard             1.13.1  
tensorflow-estimator    1.13.0  
tensorflow-gpu          1.13.1  
tensorflow-hub          0.5.0   
```
