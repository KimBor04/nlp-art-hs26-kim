## Expected Test Accuracy

- Random baseline: 50%
- Architecture 1 (fastText + MLP): 68%
- Architecture 2 (fastText + RNN): 75%

### Reasoning

I expect the RNN-based model to outperform the MLP because
it preserves word order and can learn sequential relationships.

The MLP uses averaged word embeddings, which lose word order
and may miss important contextual information.

However, the RNN may overfit due to the relatively small
training dataset.
