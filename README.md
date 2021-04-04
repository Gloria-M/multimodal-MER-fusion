# Analysis of multimodal Music Emotion Recognition (Part II)

This repository presents the **second part** of the ***multimodal Music Emotion Recognition Analysis*** series.  
The MER task is tackled from a regeression perspective: fusions of Convolutional NN models are trained on combinations of features extracted from **audio & lyrics** and **audio & comments**, to predict values for the two dimensions of emotion: ***valence*** and ***arousal***.  
  
### In the **first part**, that can be found [here](https://github.com/Gloria-M/multimodal-MER-simple), unimodal CNN models are separately trained on audio, lyrics and comments features
  

### For the complete description of the methods and experiments, please refer to [multimodal MER](https://gloria-m.github.io/multimodal.html).   

<br/>  

## Dataset

The dataset used is [new multimodal dataset](https://gloria-m.github.io/new_dataset.html), consisting in audio, lyrics and listeners' comments, with different types of annotations available, including scores in the 2D space of the emotions defined in [[Russell, 1980]](https://www.researchgate.net/publication/235361517_A_Circumplex_Model_of_Affect), represented by ***valence*** and ***arousal*** dimensions.  
  
### Data path structure

The data directory should have the following structure:
```
.
├── Data
    ├── Audio
    │   ├── *.mp3
    ├── Comments
    ├── Lyrics
    ├── annotations.json
    ├── comments.json
    ├── lyrics.json
```  

## Usage  

### 1. Prepare data

#### run `python main.py --mode=preprocess`  

> Extract annotations and data info from `annotations.json`    
> Augment dataset
> Make train, validation and test sets
> Extract MFCC features from waveforms  
> Lyrics & Comments Word2Vec embeddings  
  
Control the preprocessing method by modifying the default values for the following parameters:
```
--samples_per_quadrant = 3000 (number of samples in each of the four quadrants after augmentation)  
--train_val_test_ratio = [0.7, 0.15, 0.15] (train, validation and test sets size. NOTE: must sum up to 1)
```  

### 2. Train

#### run `python main.py --text_modality=*`  

There are two options for training:  
 - audio & lyrics model: `--text_modality=lyrics` will create and train a model using the MFCCs audio features   
 - audio & comments model: `--text_modality=comments` will create and train a model using the comments features   
  
The pre-trained models created previously in [Part I](https://github.com/Gloria-M/multimodal-MER-simple) can be loaded using `--load_pretrained=true`.  
This will load the weights of the convolutional layers and freeze them. Only the final part of the model consisting in two linear layers will be trained.
  
Control the training by modifying the default values for the following parameters:
```
--device = cuda (train on cuda)  
--log_interval = 1 (print train & validation loss each epoch)
--num_epochs = 2000
```  
  
> The trained model will be saved at `Models/audio_<text_modality>_model.pt`  

### 3. Test

#### run `python main.py --mode=test --modality=*`  
  
> The model saved as `Models/audio_<text_modality>_model.pt` will be loaded.   
> - audio & lyrics model: `--text_modality=audio`   
> - audio & comments model: `--text_modality=comments`  

<br/>  

### Tools  
`PyTorch`, `librosa`, `NLTK`, `Word2Vec`
