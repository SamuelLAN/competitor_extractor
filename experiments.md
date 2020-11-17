(baseline) fixed mean sentence embedding + MLP

fixed sentence embedding sequence + MLP

fixed sentence embedding sequence + LSTM

fixed sentence embedding sequence + Transformer

...

fine-tuned sentence embedding ...

...

Combine other features ...

...

Other BERT ...

...

data augmentation, self-supervision (chunk some of the descriptions)

...

multi-task
    predict industry
    predict competitor

or use the model for the prediction of industry to extract the features, and then add it to the description embeddings

...

adversarial training

    add noise to the embeddings
