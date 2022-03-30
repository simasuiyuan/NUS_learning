#%%
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import json
import nltk
from tqdm import tqdm
from pathlib import Path
tqdm.pandas()
# nltk.download('stopwords')
# nltk.download('punkt')
%matplotlib inline
IMAGE_PATH = Path("/etlstage/PEE_joint/mine/image_data")
# %%
nltk.data.path.append("/home/hdfsf10n/nltk_data")
nltk.data.path
# %%
%load_ext autoreload
%autoreload
from src.utils.data_loader import get_loader
# %%
with open('./data/vaild_dataset.json', 'r') as outfile:
  product_info = pd.read_json(json.load(outfile), orient="records")
  
product_info["file_loc"] = product_info["file_name"].apply(lambda file_name : IMAGE_PATH/file_name)
product_info

product_info.head()
# %%
product_info["perCategory"].unique()
# %%
idex = 10
# img_url = product_info.loc[idex,"imageURLHighRes"]
img_loc = product_info.loc[idex,"file_loc"]
caption = product_info.loc[idex,"description"]
Category = product_info.loc[idex,"perCategory"]
print(caption)
print(Category)
print(img_loc)
I = io.imread(img_loc)
plt.axis('off')
plt.imshow(I)
# %%
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder

image_dict = product_info["file_loc"].to_dict()
caption_dict = product_info["description"].to_dict()
category_dict = product_info["perCategory"].to_dict()
onehot_cat = OneHotEncoder().fit_transform(np.array([*category_dict.values()], dtype=object).reshape(-1, 1)).toarray()

# Define a transform to pre-process the training images.
# transform_train = transforms.Compose([ 
#     transforms.Resize(500),                          # smaller edge of image resized to 256
#     transforms.Resize((320,320)),                      # get 224x224 crop from random location
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])
transform_train = transforms.Compose([ 
    transforms.Resize(500),                          # smaller edge of image resized to 256
    transforms.Resize((320,320)),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
])

# Set the minimum word count threshold.
vocab_threshold = 5

mode = "train"

# Specify the batch size.
# we will pass 10 images at a time. So, m = 10
batch_size = 10
# %%
%load_ext autoreload
%autoreload
from src.utils.data_loader import get_loader
# %%
data_loader = get_loader(product_info,
                        transform=transform_train,
                        mode='train',
                        image_type="file_loc",
                        batch_size=batch_size,
                        vocab_threshold=vocab_threshold,
                        vocab_from_file=False)

print('The shape of first image:', data_loader.dataset[0][0].shape)
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader.dataset.get_indices()
print('sampled indices:', indices)

# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler
    
# Obtain the batch.
images, onehot_cat, captions = next(iter(data_loader))
    
print('images.shape:', images.shape)
print('onehot_cat.shape:', onehot_cat.shape)
print('captions.shape:', captions.shape)
# %%
plt.axis('off')
plt.imshow(images[0].detach().T)
plt.show()
# %%
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import sys
import os
import math
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import nltk
from nltk.translate.bleu_score import corpus_bleu

%load_ext autoreload
%autoreload
from src.utils.data_loader import get_loader
from src.models.CNN_Encoder import EncoderCNN
from src.models.RNN_Decoder import DecoderRNN
from src.models.MLP_Encoder import MlpEncoder
from src.utils.utils_trainer import get_batch_caps, get_hypothesis, adjust_learning_rate
# %%
def train(epoch, 
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion, 
          total_step, 
          num_epochs, 
          data_loader, 
          write_file, 
          save_every = 1):
    """ Train function for a single epoch. 
    Arguments: 
    ----------
    - epoch - number of current epoch
    - encoder - model's Encoder
    - decoder - model's Decoder
    - optimizer - model's optimizer (Adam in our case)
    - criterion - loss function to optimize
    - num_epochs - total number of epochs
    - data_loader - specified data loader (for training, validation or test)
    - write_file - file to write the training logs
    
    """
    epoch_loss = 0.0
    epoch_perplex = 0.0
    
    for i_step in range(1, total_step+1):
        # training mode on
        encoder.train() # no fine-tuning for Encoder
        # mlp.train()
        decoder.train()
        
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, onehot_cat, captions = next(iter(data_loader))
        # target captions, excluding the first word
        captions_target = captions[:, 1:].to(device) 
        # captions for training without the last word
        captions_train = captions[:, :-1].to(device)

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)

        onehot_cat = onehot_cat.type('torch.FloatTensor').view(batch_size,-1).to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        # mlp.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)

        outputs, atten_weights = decoder(captions= captions_train,
                                         features = features,
                                         category_enc = onehot_cat)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions_target.reshape(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        perplex = np.exp(loss.item())
        epoch_loss += loss.item()
        epoch_perplex += perplex
        
        stats = 'Epoch train: [%d/%d], Step train: [%d/%d], Loss train: %.4f, Perplexity train: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), perplex)
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        write_file.write(stats + '\n')
        write_file.flush()
        
        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)
        
    epoch_loss_avg = epoch_loss / total_step
    epoch_perp_avg = epoch_perplex / total_step
    
    print('\r')
    print('Epoch train:', epoch)
    print('\r' + 'Avg. Loss train: %.4f, Avg. Perplexity train: %5.4f' % (epoch_loss_avg, epoch_perp_avg), end="")
    print('\r')
    
    # Save the weights.
    # if epoch % save_every == 0:
    #     torch.save(decoder.state_dict(), os.path.join('./models_new', 'decoder-%d.pkl' % epoch))
    #     torch.save(encoder.state_dict(), os.path.join('./models_new', 'encoder-%d.pkl' % epoch))
# %%
def validate(epoch, 
             encoder, 
             decoder, 
             encoder_optimizer, 
             decoder_optimizer, 
             criterion, 
             total_step, num_epochs, data_loader, write_file, bleu_score_file):
    """ Validation function for a single epoch. 
    Arguments: 
    ----------
    - epoch - number of current epoch
    - encoder - model's Encoder (evaluation)
    - decoder - model's Decoder (evaluation)
    - optimizer - model's optimizer (Adam in our case)
    - criterion - optimized loss function
    - num_epochs - total number of epochs
    - data_loader - specified data loader (for training, validation or test)
    - write_file - file to write the validation logs
    """
    epoch_loss = 0.0
    epoch_perplex = 0.0
    references = []
    hypothesis = []
      
    for i_step in range(1, total_step+1):
        # evaluation of encoder and decoder
        encoder.eval()
        decoder.eval()
        val_images, val_onehot_cat, val_captions, caps_all = next(iter(data_loader))
        
        val_captions_target = val_captions[:, 1:].to(device) 
        val_captions = val_captions[:, :-1].to(device)
        val_images = val_images.to(device)
        val_onehot_cat = val_onehot_cat.type('torch.FloatTensor').view(batch_size,-1).to(device)
        
        features_val = encoder(val_images)
        outputs_val, atten_weights_val = decoder(captions= val_captions,
                                                 features = features_val,
                                                 category_enc=val_onehot_cat)
        loss_val = criterion(outputs_val.view(-1, vocab_size), 
                             val_captions_target.reshape(-1))
        
        # preprocess captions and add them to the list
        caps_processed = get_batch_caps(caps_all, batch_size=batch_size)
        references.append(caps_processed)
        # get corresponding indicies from predictions
        # and form hypothesis from output
        terms_idx = torch.max(outputs_val, dim=2)[1]
        hyp_list = get_hypothesis(terms_idx, data_loader=data_loader)
        hypothesis.append(hyp_list)
        
        perplex = np.exp(loss_val.item())
        epoch_loss += loss_val.item()
        epoch_perplex += perplex
        
        stats = 'Epoch valid: [%d/%d], Step valid: [%d/%d], Loss valid: %.4f, Perplexity valid: %5.4f' % (epoch, num_epochs, i_step, total_step, loss_val.item(), perplex)
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        write_file.write(stats + '\n')
        write_file.flush()
        
        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)
    
    epoch_loss_avg = epoch_loss / total_step
    epoch_perp_avg = epoch_perplex / total_step
            
    # prepare the proper shape for computing BLEU scores
    references = np.array(references).reshape(total_step*batch_size, -1)
    #hyps = np.array(hypothesis).reshape(total_step*batch_size, -1)
    hyps = np.concatenate(np.array(hypothesis))
        
    bleu_1 = corpus_bleu(references, hyps, weights = (1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hyps, weights = (0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hyps, weights = (1.0/3.0, 1.0/3.0, 1.0/3.0, 0))
    bleu_4 = corpus_bleu(references, hyps, weights = (0.25, 0.25, 0.25, 0.25))
    # append individual n_gram scores
    #bleu_score_list.append((bleu_1, bleu_2, bleu_3, bleu_4))
    
    print('\r')
    print('Epoch valid:', epoch)
    epoch_stat = 'Avg. Loss valid: %.4f, Avg. Perplexity valid: %5.4f, \
    BLEU-1: %.2f, BLEU-2: %.2f, BLEU-3: %.2f, BLEU-4: %.2f' % (epoch_loss_avg, epoch_perp_avg, bleu_1, bleu_2, bleu_3, bleu_4)
    
    print('\r' + epoch_stat, end="")
    print('\r')
    
    bleu_score_file.write(epoch_stat + '\n')
    bleu_score_file.flush()
    return bleu_1, bleu_2, bleu_3, bleu_4
# %%
batch_size = 10          # batch size, change to 64
vocab_threshold = 3        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 125           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_features = 2048        # number of feature maps, produced by Encoder
num_epochs = 200               # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss

log_train = './logs/training_log.txt'       # name of files with saved training loss and perplexity
log_val = './logs/validation_log.txt'
bleu = './logs/bleu.txt'
# %%
train_df, test_df = train_test_split(product_info, test_size=0.2)
train_df.reset_index(drop=True,inplace=True)
test_df.reset_index(drop=True,inplace=True)


# transform_train = transforms.Compose([ 
#     transforms.Resize(500),                          # smaller edge of image resized to 256
#     transforms.Resize((320,320)),                      # get 224x224 crop from random location
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])
transform_train = transforms.Compose([ 
    transforms.Resize(500),                          # smaller edge of image resized to 256
    transforms.Resize((320,320)),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
])

# Build data loader.
data_loader = get_loader(train_df,
                         transform=transform_train,
                         mode='train',
                         image_type="file_loc",
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Setup the transforms
# transform_test = transforms.Compose([ 
#     transforms.Resize((320,320)),                   # smaller edge of image resized to 500
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])
transform_test = transforms.Compose([ 
    transforms.Resize((320,320)),                   # smaller edge of image resized to 500
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
])

# Create the data loader.

valid_data_loader = get_loader(test_df,
                               transform=transform_train,
                               mode='valid',
                               image_type="file_loc",
                               batch_size=batch_size)


total_step_valid = math.ceil(len(valid_data_loader.dataset.caption_lengths) / valid_data_loader.batch_sampler.batch_size)
total_step_valid
# %%
# Initialize the encoder and decoder. 
encoder = EncoderCNN()
decoder = DecoderRNN(num_features = num_features, 
                     embedding_dim = embed_size, 
                     category_dim = len(set(category_dict.values())),
                     hidden_dim = hidden_size, 
                     vocab_size = vocab_size)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)
# %%
print(device)
# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

#params = list(decoder.parameters()) + list(encoder.parameters()) 
params = list(decoder.parameters())

# TODO #4: Define the optimizer.
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper

encoder_lr = 1e-3  # learning rate for encoder if fine-tuning
encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),lr=encoder_lr)
encoder_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(encoder_optimizer, 
                                                         base_lr=encoder_lr/10, max_lr=encoder_lr*10, 
                                                         step_size_up = 10 , step_size_down=20, cycle_momentum=False)


decoder_lr = 1e-3  # learning rate for decoder
decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),lr=decoder_lr)
decoder_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(decoder_optimizer, 
                                                         base_lr=decoder_lr/10, max_lr=decoder_lr*10, 
                                                         step_size_up = 10 , step_size_down=20, cycle_momentum=False)

fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
# %%
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
if not os.path.exists(f'/etlstage/PEE_joint/mine/models_new/{current_time}'):
  os.makedirs(f'/etlstage/PEE_joint/mine/models_new/{current_time}')

# Open the training log file.
file_train = open(log_train, 'w')
file_val = open(log_val, 'w')
bleu_score_file = open(bleu, 'w')

# store BLEU scores in list 
bleu_scores = []
total_step_valid = math.ceil(len(valid_data_loader.dataset.caption_lengths) / valid_data_loader.batch_sampler.batch_size)
epochs_since_improvement = 0
best_bleu = 0.

for epoch in range(0, num_epochs+1):
    # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
    if epochs_since_improvement == 30:
        break
    # if epochs_since_improvement > 0 and epochs_since_improvement % 20 == 0:
    #     adjust_learning_rate(decoder_optimizer, 0.8)
    #     if fine_tune_encoder:
    #         adjust_learning_rate(encoder_optimizer, 0.8)

    train(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, total_step, num_epochs =num_epochs,
          data_loader = data_loader,
          write_file = file_train, 
          save_every = 1)
    
    bleus= validate(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
                                             total_step = total_step_valid, 
                                             num_epochs = num_epochs, 
                                             data_loader = valid_data_loader, write_file=file_val, bleu_score_file=bleu_score_file)
    
    encoder_lr_scheduler.step()
    decoder_lr_scheduler.step()
    
    # Check if there was an improvement
    is_best = sum(bleus)/4 > best_bleu
    best_bleu = max(sum(bleus)/4, best_bleu)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        torch.save(decoder.state_dict(), os.path.join(f'/etlstage/PEE_joint/mine/models_new/{current_time}', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join(f'/etlstage/PEE_joint/mine/models_new/{current_time}', 'encoder-%d.pkl' % epoch))
        epochs_since_improvement = 0
    
file_train.close()
file_val.close()
bleu_score_file.close()
# %%
# Setup the transforms
# transform_test = transforms.Compose([ 
#     transforms.Resize((320,320)),                   # smaller edge of image resized to 500
#     transforms.ToTensor(),                           # convert the PIL Image to a tensor
#     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])

transform_test = transforms.Compose([ 
    transforms.Resize(500),                          # smaller edge of image resized to 256
    transforms.Resize((320,320)),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
])

test_data_loader = get_loader(test_df,
                              image_type="file_loc",
                              transform=transform_test,
                              batch_size=1,
                              mode='test')
# %%
current_time = "2022-03-30-21-06-21"
encoder_file = 'encoder-27.pkl' 
decoder_file = 'decoder-27.pkl'

embed_size = 125           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_features = 2048        # number of feature maps, produced by Encoder

# The size of the vocabulary.
vocab_size = len(test_data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN()
encoder.eval()
decoder = DecoderRNN(num_features = num_features, 
                     embedding_dim = embed_size,
                     category_dim = len(set(category_dict.values())),
                     hidden_dim = hidden_size, 
                     vocab_size = vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join(f'/etlstage/PEE_joint/mine/models_new/{current_time}', encoder_file), map_location='cpu'))
decoder.load_state_dict(torch.load(os.path.join(f'/etlstage/PEE_joint/mine/models_new/{current_time}', decoder_file), map_location='cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)
# %%
def clean_sentence(output, data_loader):
    vocab = data_loader.dataset.vocab.idx2word
    words = [vocab.get(idx) for idx in output]
    words = [word for word in words if word not in (',', '.', '<end>')]
    sentence = " ".join(words)
    
    return sentence

def get_prediction(data_loader):
    orig_image, image, cat_onehot, caption= next(iter(data_loader))
    plt.imshow(orig_image.squeeze())
    # print(cat_onehot)
    plt.title(caption)
    plt.show()
    image = image.to(device)
    cat_onehot = cat_onehot.type('torch.FloatTensor').view(1,-1).to(device)
    features = encoder(image)
    output, atten_weights = decoder.greedy_search(features, cat_onehot)    
    sentence = clean_sentence(output,data_loader)
    print(sentence)
# %%
get_prediction(test_data_loader)
# %%

# %%
