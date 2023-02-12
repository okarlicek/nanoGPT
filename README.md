# nanoGPT

Implementation of generating transformer inspired by GPT following [video from Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY). 
This implementation uses simple character-level embeddings and is trained on the Shakespeare dataset. 
The size of the model was very limited by my GPU (only 6 GB memory). You can see the result in the jupyter notebook. 
The generated text is not always meaningful, but it is much better than the baseline generated from the Bigram model.
Note that Karpathy uses PyTorch in his implementation, however, I used Tensorflow for two reasons.
Firstly, I am more familiar with Tensorflow, secondly, when using a different framework than the lecturer I need to write my own code and not just copy-paste.
Although, for Transformer implementation, I used some parts of code I write during [Deep Learning course from Martin Straka](https://ufal.mff.cuni.cz/courses/npfl114/2122-summer).
But, during this class, we implemented only an encoder, so this project was interesting for me to try the other side :)
