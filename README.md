# Friend-Emulator

More & more of my friends are moving away from where I live, some across different time zones.  To keep myself from becoming lonely, I figured the next best thing to physical human contact is an NLP bot that can convincingly emulate the speaking style of my friends

## Getting & processing the data

Thankfully WhatsApp allows you to export group chats as an e-mail, which provided me with 20024 unique messages with timestamps across 13 of my friends as a text file.  Unfortunately, it needed a little preprocessing to get into a suitable format for analysis

As I'm not one to re-invent the wheel here, journocode has published an [amazing script for parsing the raw text data into a CSV file,](https://github.com/journocode/datavizwhatsapp) and with a bit of tinkering I was able to get the messages in a nice format for analysis with pandas and seaborn

## Exploratory Data Analysis

First off, a click visualisation of user activity

![User Activity](https://github.com/MetcalfeTom/Friend-Emulator/blob/master/UserPieChart.png)

# The Neural Networks

I considered separating the messages by user and feeding that information as a 13-dimensional one-hot vector, [but character-level RNNs have been shown to match the format style of their training sets with surprising accuracy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (and this makes for a better experiment anyway), so other than removing the time stamps and media messages, I left the chat log as is

I wanted to test out the effectiveness of character-level modelling vs using word embeddings, as well as figure out how best to optimize the training time given my (relatively) underpowered GPU, so for the first few iterations I held running time for thirty epochs as my metric, given a batch size of 32

## Iteration 1:
#### Character-level 3-layer network, runtime: 4.5 hours
```metcalfe: we slept under sool pay it the part in should it
hyde: i'm anyone was going that a the the now
hackney: i'm good
hackney: i was lang is for the and deally have a the at we be on the day and bare the can be longand do it was it over to mace it could come to the mare of it i think i the getter to we antter tonight then?
hackney: i'm were probably think the are bes lave bry wetterds my let have
```

I ran a 3-layer neural network: two 128-unit LSTM layers (with 50% dropout) and then a fully-connected 32-unit dense layer.  The network took in sequences of 50 characters from the chat log and was trained to predict the next character.  Using a step size of 7 characters I was able to produce 126000 sequences for training.  Gradient descent wasn't particularly fast, however I did learn a few things:

- Even after the first 3 epochs, the model began to replicate the format of the conversation quite well: it used line breaks and then began each new line with a username and colon (although it did "invent" some of its own names to begin with).  This is due to the sequence of line breaks followed by users being so frequent in the training set.
- I wanted to leave emoji in to begin with, but the model rarely used them because each emoji is treated as its own character, so the probability of any of them being used after a standard character is too small
- With a low diversity, the predictions favoured long unbroken messages over short ones, and overused common words like "the", "and" or "to"
- It learned very quickly how verbose Hackney could be

## Iteration 2:
#### Character-level 2-layer network, runtime: 1 hour
```metcalfe: just post sty
hackney: what wy the wiel
metcalfe: i thinka?
orr: and you just on ersice somnhel did one aboet ha pay in sare this outd
joyce: yeah you thitk no one going
denison: i'm mucka
denison: it's the coulds tromans?
joyce: yeah it maybe araninginge
hackney: creds?
orr: ha're time
joyce: i rangurang a out for me
hyde: ghing odes
denison: moleime me  
```
I took out the second LSTM layer in my network to lower the runtime and used RMSProp for optimization.  I filtered out all but the 70 most common characters this time, which in turn meant I had to also remove blank messages, trimming my training set to 125500 sequences.

The system ran 4x faster, but qualitatively it made a lot of errors in its prediction, so I'll shoot for somewhere in between.

## Iteration 3:
#### Character-level 3-layer network, run time: 2 hours

```
hackney: what a that the looking it do the metter the port to me on a pore is not a sell that tame the thought as i gonna a thinking the part of the warn
tromans: i was not the man
hyde: something to the same the faride the powe work are you really have a tains on a nem playing that to cond
denison: when they tains in the price
hackney: so we think the poon they was anyone was it was
```

I went back to the 3-layer configuartion of my first network, but doubled the learning rate to speed up gradient descent.  Once again limited the character set to 70 and used RMSProp.

I've managed to double my algorithm's efficiency, but the model is failing to fit well on the training set, so I'll try to correct the bias by adding more layers

## Iteration 4:
#### Character-level 5-layer network

One thing I'm unhapy about though, is that the model is playing it safe and sticking to the most common words in the chat vocabulary, and not making use of any of the diverse and coveted slang.

Wanted to take same architecture and apply to word embeddings, however long-term dependency has now changed significantly since the messages are held in shorter sequences.  The information stored is now more likely to take into account replies between different users
