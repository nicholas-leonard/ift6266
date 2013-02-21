ift6266
=======

Emotion Recognition

This model integrates with my pylear2 fork.

Feb 14 2013
-----------

Tried one Tanh hidden layer, but decided to try out RectifiedLinear hidden
layers with dropout. Had some difficulty determining the learning rate, as I
am used to dealing with normalized bag of words, which in my experience 
work well with 0.1. Gabriel Bernier-Colborne's blog entry helped me out:
http://ift6266gbc.wordpress.com/2013/02/14/first-attempt/ .

I forked a copy of the make_submission.py to make it work without yaml.

Feb 21 2013
-----------
Implemented a database schema that maps pylearn2 classes to SQL Tables and 
pylearn2 class instances to rows of these Tables. Instances can be combined
into leanring algorithms where each such configuration represents a 
particular MLP layout with its hyperparameters.

A python program was implemented to read configurations from the database
and train a model for each one, storing results in the database for later 
analysis. Each launched process executing this program basically loops over
untried model configurations found in the database.

Results thus far are presented in ./results1.csv. They are ordered from 
highest classification accuracy to lowest. None of these results make use
of weight decay. The first two layers of each configuration are shown. As
you can see, the two hidden layer networks perform worse than one hidden 
layer networks (for now). As experimented by Pier-Luc Carrier 
(http://plcift6266.wordpress.com/2013/02/18/experiment-2-single-hidden-layer-mlp/), 
I started from a small hidden layer with 500 rectified linear units. Then I 
applied dropout to the input of each layer as mentionned in Hinton 2012 
http://arxiv.org/pdf/1207.0580.pdf (0.2 in first layer, 0.5 in remainder).
With this, more hidden units seem to be more appropriate. For now my largest
layer has 2500 units and provides the best results. I have queued 
configurations with more units. Learning rates of 0.0001 and 0.001 seem to 
provide better results than my first post mentionned, but this is due to the
use of the Standardize preprocessing object, which is now used by default.

I haven't tried weight decay just yet, but I am waiting on results for such 
configurations (they are in the queue). 

I wan't to try generating training examples using: 
Translation, rotation, scaling, affine transformation, mirror. 