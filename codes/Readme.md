# Do Deep Neural Networks Capture Emotion-Color Association?

All of the codes were written in Jupyter Notebook. So we are attaching all the Jupyter Notebooks file which we used to execute our code. We are also providing one single python module which can be used evaluate important analysis. Note that outputs could vary from the paper but it will remain under the mean and std which we reported in the paper. We did not add random seed because we are already reporting the mean and variance over 50 runs. For exact run which we used you can check the executed ipynb notebook files. 

**Requirements:**
- Python 3
- Tensorflow 2
- Jupyter Notebook
- Keras
- tqdm
- matplotlib
- scipy
- numpy

You can simply run these notebook files on Google Colab (Which we used to run our codes). All of the packages are already available there. But you will need to upload the data files. We have removed the automated download link because of "Double Blind Submission". Also note that all of the notebook files are executed one so you can directly head on to check the results without even running the whole program.

---

## Following are the legends to different files.

### For quick evaluation of important results run following two python files:
- important_results/eval_classification.py  {Takes ~ 11 mins on Nvidia Tesla T4}
- important_results/eval_correlation.py  {Takes ~ 8 mins on Nvidia Tesla T4}

### For Correlation Scores
- correlation/VGG16.ipynb - VGG16 correlation scores for transformed representations.
- correlation/VGG16_Raw.ipynb - VGG16 correlation scores for raw representations.
- correlation/VGG16_With_Wrong_Class.ipynb - VGG16 correlation scores for wrong colored labels.
- Rest are for calculating correlation scores for other models used in our paper.

### For Classification Task
- classification/Color_Similarity.ipynb - Few Shot emotion classification task using our color-similarity based decision model.
- classification/Standard.ipynb - Standard cross-entropy based classification model.
- classification/Human_And_Chance.ipynb - Random iteration to evaluate chance accuracy.

---

### Demo outputs for eval_classification.py and eval_correlation.py

#### eval_classification.py
Training Color_Similarity model >>>>>>
2/2 [==============================] - 1s 466ms/step
1/1 [==============================] - 0s 7ms/step
100%
50/50 [06:08<00:00, 7.38s/it]


Training Standard cross entropy model >>>>>>
2/2 [==============================] - ETA: 0sWARNING:tensorflow:Callbacks method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0040s vs `on_predict_batch_end` time: 0.1228s). Check your callbacks.
2/2 [==============================] - 0s 139ms/step
100%
50/50 [02:38<00:00, 3.17s/it]

100%
50/50 [01:35<00:00, 1.91s/it]

Raw Similarity | wrt Human Prediction - 40 | wrt Actual Class - 24
Transformed Similarity | wrt Human Prediction - 56.12 | wrt Actual Class - 39.56
Standard Classification (Trained on Human Prediction) | wrt Human Prediction - 43.72 | wrt Actual Class - 33.12
Standard Classification (Trained on Actual Class) | wrt Human Prediction - 33.0 | wrt Actual Class - 30.2

Total eval time: 638 secs


#### eval_correlation.py
Estimating Raw Correlation Score on VGG16 >>>>>>
100%
1/1 [00:31<00:00, 31.99s/it]
2/2 [==============================] - ETA: 0sWARNING:tensorflow:Callbacks method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0041s vs `on_predict_batch_end` time: 0.0867s). Check your callbacks.
2/2 [==============================] - 0s 120ms/step
1/1 [==============================] - 0s 30ms/step

Estimating Transformed Correlation Score on VGG16 >>>>>>
2/2 [==============================] - ETA: 0sWARNING:tensorflow:Callbacks method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0042s vs `on_predict_batch_end` time: 0.1108s). Check your callbacks.
2/2 [==============================] - 0s 135ms/step
1/1 [==============================] - 0s 22ms/step
100%
50/50 [06:14<00:00, 7.50s/it]

## Printing Correlation Scores on Wrong labels >>>
2/2 [==============================] - ETA: 0sWARNING:tensorflow:Callbacks method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0044s vs `on_predict_batch_end` time: 0.1102s). Check your callbacks.
2/2 [==============================] - 0s 135ms/step
1/1 [==============================] - 0s 22ms/step
### Raw Scores ####
Seq: [0, 1, 2, 3, 4] - Overall R: 0.33 and p-value: 1.3354185800653852e-07

Seq: [4, 3, 0, 2, 1] - Overall R: 0.01 and p-value: 0.9298284407429127

Seq: [4, 2, 0, 1, 3] - Overall R: -0.13 and p-value: 0.044568082230323096

Seq: [3, 4, 0, 2, 1] - Overall R: -0.01 and p-value: 0.8904583724998051

Seq: [3, 0, 1, 4, 2] - Overall R: -0.1 and p-value: 0.10962992981071425

Seq: [2, 3, 1, 4, 0] - Overall R: -0.09 and p-value: 0.16830611976766283

Seq: [2, 4, 3, 1, 0] - Overall R: 0.11 and p-value: 0.07855765020465037

Seq: [1, 0, 4, 2, 3] - Overall R: 0.06 and p-value: 0.37524274351839854

Seq: [1, 2, 3, 4, 0] - Overall R: -0.01 and p-value: 0.8630613096650089

---------------------------
### Transformed Scores ####
Seq: [0, 1, 2, 3, 4] - Overall R: 0.62 and p-value: 4.322746103146341e-28

Seq: [4, 3, 0, 2, 1] - Overall R: 0.03 and p-value: 0.5832541605772208

Seq: [4, 2, 0, 1, 3] - Overall R: -0.31 and p-value: 7.74490663793746e-07

Seq: [3, 4, 0, 2, 1] - Overall R: 0.04 and p-value: 0.5583148883596672

Seq: [3, 0, 1, 4, 2] - Overall R: -0.33 and p-value: 7.652038109422208e-08

Seq: [2, 3, 1, 4, 0] - Overall R: -0.34 and p-value: 4.2759493845894873e-08

Seq: [2, 4, 3, 1, 0] - Overall R: 0.04 and p-value: 0.48328120217484816

Seq: [1, 0, 4, 2, 3] - Overall R: -0.01 and p-value: 0.8601108173562265

Seq: [1, 2, 3, 4, 0] - Overall R: -0.01 and p-value: 0.8196020121793334

------------------------------------------------
## Printing Correlation Scores on Actual Labels >>>
Raw Score for fc2: 0.33
Transformed Score for fc2: mean -  0.6299279265154858 - std - 0.014911585520368035

Total eval time: 464 secs
