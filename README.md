ift6266
=======

Emotion Recognition

This model integrates with my pylear2 fork.

Feb 14 2013

Tried one Tanh hidden layer, but decided to try out RectifiedLinear hidden
layers with dropout. Had some difficulty determining the learning rate, as I
am used to dealing with normalized bag of words, which in my experience 
work well with 0.1. Gabriel Bernier-Colborne's blog entry helped me out:
http://ift6266gbc.wordpress.com/2013/02/14/first-attempt/ .

I forked a copy of the make_submission.py to make it work without yaml.