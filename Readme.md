
# Image classification for classifying cats and dogs  

Image classification of for Kaggle cats and dogs challenge based on tutorial on [Keras web blog](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).  

A ConvNet model following the tutorial at the blog. The model parameters were further saved to load the model and perform sentiment analysis.   

**Data source**: [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats/data).  

**Future prospects:**  
1) Improve model with different algorithms.  
2) Try more pre-trained image classification models.  

## Folder structure:   
1) **Root**: Main folder containing Readme and scripts.  
2) **data**: Data folder containing:   
	a) train: Training data.  
	b) validation: Test/Validation data.    
	c) test: Test data.  
3) **figures**: All generated figures from the scripts.  
4) **models**: Trained models stored in this folder.  

## Files  

**[utilities.py](https://github.com/ishmukul/ImageClassification/blob/master/utilities.py)**: Contains useful functions.  
**[ImageClassification.py](https://github.com/ishmukul/ImageClassification/blob/master/ImageClassification.py)**: ImageClassification file.  
**[CheckImages.py](https://github.com/ishmukul/ImageClassification/blob/master/CheckImages.py)**: Image augmentation example.    


## File descriptions    
=======================================================  
**[utilities.py](https://github.com/ishmukul/ImageClassification/blob/master/utilities.py)**: 
**This script is required for all other scripts.**  

*Useful common functions.*    


=======================================================  
**[ImageClassification.py](https://github.com/ishmukul/ImageClassification/blob/master/ImageClassification.py)**:   

Image classification using ConvNet neural networks.  

With simple network validation accuracy are 75%.  
Large fluctuations in accuracy history.  

Loss function and Accuracies are plotted in Figures:  
[Accuracy.png](https://github.com/ishmukul/ImageClassification/blob/master/figures/Accuracy.png)    
[Loss.png](https://github.com/ishmukul/ImageClassification/blob/master/figures/Loss.png)  
<img src="https://github.com/ishmukul/ImageClassification/blob/master/figures/Accuracy.png" alt="Accuracy" width="200"/>
<img src="https://github.com/ishmukul/ImageClassification/blob/master/figures/Loss.png" alt="Loss" width="200"/>  

