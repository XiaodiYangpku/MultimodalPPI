### MultimodalPPI ###
MultimodalPPI employ document and graph embeddings to characterize the multimodal features (i.e., sequence, network and function) of proteins to train the LightGBM models for the prediction of protein-protein interactions (PPIs) between herpes viruses and human host.

### Usage ###
# Step 0 Prepare embedding model files
  cat ./x* > embeddings.tar.gz 
  tar -zxvf embeddings.tar.gz

# Step 1 Select encoding approach (doc2vec, net, go, dpc, ct, ac) to represent proteins of samples
  cd script
  python 1_encoding.py encoding_method
  e.g. python 1_encoding.py doc2vec
  e.g. python 1_encoding.py net
  e.g. python 1_encoding.py go

# Step 2 Combine encodings (optional): concatenate the feature vectors of proteins to obtain the integrated encodings.
  cd script
  python 2_combine_multimodal.py encoding_method1 encoding_method2 encoding_method3 ...
  e.g. python 2_combine_multimodal.py doc2vec net go

# Step 3 Select machine learning algorithm (lgbm, rf, svm): train and test the model based on pre-trained embeddings by using selected algorithm.
  cd script
  python 3_train_test.py encoding_method dataset algorithm
  e.g. 'rigorous dataset' (three replicates), c3_1/c3_2/c3_3
  e.g. python 3_train_test.py doc2vecnetgo c3_1 lgbm
  e.g. python 3_train_test.py doc2vecnetgo c3_2 lgbm
  e.g. python 3_train_test.py doc2vecnetgo c3_3 lgbm

# Step 4 Assess the models of three replicates: calculate various metrics of performance of models to obtain the overall performance of the prediction method.
  cd script
  python 4_average_sd.py encoding_method dataset algorithm
  e.g. 'rigorous dataset' (three replicates), c3_1/c3_2/c3_3
  e.g. python 4_average_sd.py doc2vecnetgo c3 lgbm

### Output ###
a. Run_result.txt - performance of various methods
b. xx.model - machine learning models
c. xx.txt - prediction result files containing pairwise protein ids, labels and prediction scores.


### Notice ###
The human-virus systems and related parameters can be changed in the above python scripts.

### Dataset ###
  Two benchmarking datasets of human-herpesvirus PPI system. The ratio of positive-to-negative is 1:10. Three replicates were set for each bencmarking dataset.
  'non-rigorous dataset', sample/1_10_c1_1/; sample/1_10_c1_2/; sample/1_10_c1_3/
  'rigorous dataset', sample/1_10_c3_1/; sample/1_10_c3_2/; sample/1_10_c3_3/
  9,439 human-herpesvirus PPIs, 94,390 non-human-Herpes PPIs.
  
### Requirements ###
  - Keras (==2.2.4)
  - scikit-learn (==0.22.1)
  - numpy (==1.16.6)

### Citation ###
Please kindly cite the paper if you use refers to the paper, code or datasets.
@article{Yang2023MultimodalPPI,
  title={Multi-modal features-based human-herpesvirus protein-protein interaction prediction by using LightGBM},
  author={Yang, Xiaodi and Wuchty, Stefan and Liang, Zeyin and Ji, Li and Wang, Bingjie and Zhu, Jialin and Zhang, Ziding and Dong, Yujun},
  journal={xx}
}


