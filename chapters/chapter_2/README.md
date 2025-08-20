### N-gram and Corpus-based Techniques

* N-gram models improve on Bag-of-Words by introducing context through N, allowing word generation based on previous tokens. Trigrams (N=3) and other lengths (up to pentagrams or septagrams) can be used for flexibility.
* Text is cleaned and padded minimally to help the model, then trained using everygrams for flexibility over efficiency.
* At generation, the model can use a few initial tokens to predict the next words.
* N-gram models do not attempt full linguistic modeling—they ignore syntax and semantic meaning, focusing only on probabilistic connections between words in an N-length phrase.
* Despite their simplicity, they are useful for practical applications and provide strong baselines for text analysis.
* N-grams rely on static cues (whitespace, orthography) and assume fixed phrase lengths, giving quick insights if the context is known.
* They fail to capture the semantic meaning of words, which can be addressed using Bayesian language modeling.


### Bayesian Techniques

* Bayes’ theorem provides a simple, mathematically sound framework to calculate the probability of a hypothesis given evidence:

$$
P(\text{hypothesis | evidence}) = \frac{P(\text{evidence | hypothesis}) \times P(\text{hypothesis})}{P(\text{evidence})}
$$

* In language modeling, it can be applied to tasks like sentiment classification, where the model estimates the likelihood of a label given the words in a sentence.
* Bayesian models assign static probabilities to words, which can fail for ambiguous or context-dependent words like pronouns. For instance, words such as “it” receive fixed LogPrior and LogLikelihood values, which may misrepresent their actual usage.
* These models are not generative—they cannot produce new sentences—but they excel at classification and can augment generative models.
* A naive Bayes classifier uses a frequency-based approach to compute log priors and log likelihoods for each word-label pair, enabling predictions by summing these logs.
* Training involves counting word occurrences per label, calculating probabilities with Laplace smoothing, and computing log ratios for classification.
* Bayesian language models ignore word order, treating sequences like a bag-of-words, making them less suited for modeling context or generating language.
* Despite their theoretical appeal, Bayesian models are often overhyped for practical classification tasks. Their main strength is probabilistic reasoning, not capturing the structure or semantics of sequences.

### Markov Chains

* **Markov chains** (or HMMs) extend N-gram models by adding **state**: the next word depends only on the current word.

$$
P(X_{t+1} \mid X_1, \dots, X_t) = P(X_{t+1} \mid X_t)
$$

* Common uses:

  * **PoS tagging** – label words with their part of speech.
  * **NER** – identify entities like `"LA"` → `"Los Angeles"` → `"City"`.

* Key idea: stochasticity matters. Probability of the next word adjusts based on the current word, e.g., `"happy"` is unlikely immediately after `"happy"` but more likely after `"am"`.

* **Text generation**:

  * Tokenize input (e.g., whitespace).
  * Build a dictionary mapping each token → list of possible next tokens.
  * Generate text by randomly choosing the next word from this list.

* **Pros:** fast, intuitive, good for predictive text, descriptive modeling.

* **Cons:**

  * No long-term context.
  * Semantically limited (produces syntactically correct but meaningless sentences).
  * Context beyond “now” requires embedding-based or continuous models.

* Historical note: Markov initially modeled independent continuous states, then vowels in literature. HMMs remain a foundation in sequence modeling.

### Continuous Bag-of-Words (CBoW)

* **CBoW** is a predictive language model that learns word embeddings by trying to **guess a missing word** from its surrounding context.

* Unlike classical bag-of-words, it **considers context** rather than just counting words, using a “fill-in-the-blank” approach.

* Context window: selects words around a target word (before and after).

  * Example: `"Learning about linguistics makes me happy"`
  * Target = `"linguistics"`, Context = `["learning", "about", "makes", "me"]`

* Words are represented as **dense vectors** rather than one-hot encodings, enabling semantic understanding.

* Input: average vector of context words → hidden layer → output probabilities over vocabulary → predicted target.

* Learns embeddings by adjusting vectors during training via backpropagation.

* Can capture **semantic similarity**: words appearing in similar contexts get similar embeddings.

* Local context only: works well for short-range relationships but doesn’t model full sentences sequentially.

* Uses **softmax** at output to assign probabilities to every word in the vocabulary.

* Training loop: feed context → predict target → compute loss → backpropagate → update embeddings and weights.

* ReLU or other activations often used in the hidden layer to learn meaningful latent features.

* Lightweight and fast compared to sequential models (like RNNs) but less suited for generative tasks requiring long-term dependencies.

* Main strength: creating **high-quality word embeddings** that can be reused in other NLP tasks (classification, clustering, similarity searches).

### Embeddings

* Embeddings are **continuous vector representations of words** that inject meaning into tokenized text, moving beyond simple one-hot or frequency-based encodings.

* They learn meaning from **patterns in usage**, primarily **collocation**—words appearing near each other—but also capture subtle semantic relationships.

* Word2Vec popularized embeddings with the **CBoW and Skip-Gram** architectures, producing vectors that encode analogies, e.g.:

  * `"king" - "man" + "woman" ≈ "queen"`
  * This reflects human-like semantic reasoning, though embeddings lack **pragmatic understanding**. Humans consider context beyond mere word usage, whereas embeddings are purely statistical.

* Embeddings are updated during training by observing contexts, moving semantically similar words **closer together in vector space**.

* Visualization is a useful tool to **inspect embeddings**:

  * Extract vectors for a set of words.
  * Reduce dimensions (PCA, t-SNE, UMAP).
  * Plot to check if semantically related words cluster.
  * Sparse embeddings can already reveal meaningful patterns, even with simple models like CBoW.

* Advantages:

  * Encode **semantic similarity**, analogies, and syntactic relationships.
  * Provide a foundation for **downstream tasks** like classification, translation, or question answering.

* Limitations:

  * Embeddings are **context-independent** (the same vector for a word in all senses).
  * They ignore **pragmatics** and real-world knowledge beyond observed text.

* The field continues to evolve: denser embeddings, richer contextual models, and **chain-of-thought** or instructive approaches aim to combine semantic understanding with broader reasoning capabilities.

* Embedding analysis and visualization remain key for **model explainability** and **pretraining validation**, ensuring semantically related words behave as expected in vector space.

### Multilayer Perceptrons (MLPs)

* MLPs are essentially **a collection of simple “feature detectors”** bound together to recognize more complex patterns.

  * Each weight and bias in a neuron specializes in detecting something small or simple.
  * Layering many neurons allows the network to detect **hierarchical, higher-order features**.

* They form the **core building block of most neural networks**:

  * Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) still rely on fully connected layers; the main differences are how data flows and is processed.
  * The fully connected layers themselves behave the same way across architectures—they just get fed different representations (spatial in CNNs, sequential in RNNs).

* MLPs are flexible:

  * Can have **any number of hidden layers or neurons per layer**.
  * Can use different **activation functions** like Sigmoid, ReLU, or Tanh.
  * Output layer can be adapted for regression, classification, or other tasks.

* Forward pass workflow:

  * Input data flows through each linear layer.
  * Nonlinear activation is applied after each hidden layer.
  * Final layer produces output, optionally passing through softmax for classification.

* Key insights:

  * Size isn’t everything—just stacking millions of layers doesn’t guarantee better performance.
  * The **magic in modern LLMs** isn’t only in depth or number of parameters, but in **how tokenized embeddings are fed, transformed, and interacted with** across layers.
  * MLPs are deterministic once configured, but their flexibility allows them to scale into complex architectures like deep feedforward networks.

* MLPs vs older models:

  * Unlike the fixed two-layer CBoW models, MLPs are **dynamic in design**, adjustable to task needs.
  * The principles of MLPs underpin almost all neural models: even RNNs and transformers ultimately rely on repeated applications of fully connected transformations combined with clever handling of data structure (sequences, attention, convolutions, etc.).

* Conceptually, MLPs are the **foundation of “stacking simple learners to build complex understanding”**, a philosophy that echoes throughout modern deep learning.

Perfect! Here’s your RNN/LSTM section rewritten in the same bullet style:

### Recurrent Neural Networks (RNNs) and Long Short-Term Memory Networks (LSTMs)

* **RNNs are designed to process sequences**:

  * A sequence is an ordered array of elements; changing the order changes the meaning.
  * Unlike traditional MLPs, RNNs maintain an internal **state (memory)** that updates as new inputs are processed.
  * Connections can form cycles, allowing information to persist across steps.

* **Advantages over previous models**:

  * Can handle **variable-length inputs**, unlike static-input models.
  * Can capture dependencies between elements in a sequence, unlike simple Markov models.

* **Challenges with RNNs**:

  * **Vanishing gradient problem**: Gradients shrink as sequences get longer, making the model forget important earlier information.
  * **Exploding gradients**: Gradients can also grow too large, causing unstable training.
  * Short sequences may dominate predictions, biasing the model toward shorter inputs.

* **Example of RNN bias**:

  * “I loved the movie last night” vs. “The movie I went to see last night was the very best I had ever expected to see.”
  * The shorter sentence may receive a higher predicted score due to stronger influence per word.

* **LSTMs solve the RNN limitations**:

  * Use **memory cells and gating mechanisms** to remember information over long sequences.
  * Can be **bidirectional**, processing sequences left-to-right and right-to-left.
  * Handle variable-length sequences without bias toward length.

* **Implementation notes (PyTorch example)**:

  * Embeddings are packed to efficiently handle variable-length sequences.
  * RNN class: embeds tokens, packs sequences, passes them through an RNN layer, and applies a linear layer for predictions.
  * LSTM class: adds multiple layers, bidirectionality, and dropout.
  * Forward method in LSTM: concatenates last hidden states from both directions, applies dropout, and passes through a linear layer.

* **Training setup**:

  * Accuracy is measured with a binary classification metric.
  * Optimizers: SGD for RNN, Adam for LSTM.
  * Data batches are **sorted and padded** for packed sequence processing.
  * Dataset split into training, validation, and test sets.
  * LSTM converges faster and achieves better validation performance than RNNs.

* **Key architectural differences**:

  * LSTM has multiple layers, bidirectionality, and dropout.
  * Bidirectionality improves context understanding, especially in multilingual scenarios.
  * Dropout reduces overfitting during training.

* **Limitations**:

  * Computationally expensive for long sequences.
  * Training is resource-intensive compared to MLPs.
  * Inference is moderately more expensive than simpler architectures.
  * Less suitable for very detail-specific domains without careful tuning (e.g., healthcare or law).

### Attention

* **Purpose of attention**:

  * Acts as a **mathematical shortcut** for processing larger context windows efficiently.
  * Tells the model **which parts of an input to consider** and **how much**.
  * Can be thought of as an **upgraded dictionary**: instead of just key–value pairs, a **contextual query** is added.

* **Advantages over LSTMs**:

  * Solves **slow training** issues of LSTMs.
  * Maintains **high performance with fewer epochs**.
  * Allows the model to focus on relevant parts of a sequence dynamically.

* **Types of attention**:

  * **Dot product attention**: Captures relationships between every query and every key.
  * **Bi-directional self-attention**: Queries and keys come from the **same sentence**, considering all positions.
  * **Causal attention**: Focuses only on **preceding words**, useful for language modeling.
  * **Masked attention**: Hides parts of a sequence, forcing the model to **predict missing elements**.

* **Key operations in attention**:

  * Queries (Q), Keys (K), and Values (V) are **matrix projections** of the input:

    $$
    Q = X W_Q,\quad K = X W_K,\quad V = X W_V
    $$

  * **Attention scores** = dot product of Q and K (optionally scaled):

    $$
    \text{scores} = \frac{Q K^\top}{\sqrt{d_k}}
    $$

  * **Softmax** is applied to normalize the scores:

    $$
    \alpha = \text{softmax}(\text{scores})
    $$

  * Weighted sum of V according to attention scores produces the **final attention output**:

    $$
    \text{Attention}(Q,K,V) = \alpha V
    $$

* **Intuition behind Q, K, V**:

  * Keys and Values are like a **dictionary or lookup table**.
  * Queries act as a **search** to retrieve relevant values.
  * Attention scores determine **how much focus** to give each input element.

* **Complexity and performance**:

  * Dot product attention is **quadratic** in sequence length.
  * Efficient projection into a **common space** allows scalable computation.
  * Innovations like **Hyena** and **Recurrent Memory Transformer (RMT)** aim to reduce memory and computation overhead.

* **Significance**:

  * Attention is the **core differentiator** between older NLP models (like LSTMs) and modern transformers.
  * Enables the **Transformer architecture**, which has become the standard for most state-of-the-art NLP models.
  * Balances **efficiency, scalability, and context-awareness** in sequence modeling.
 
### Summary 
| Technique                          | Key Idea                                                                 | Pros                                                                    | Cons / Limitations                                                 | Notes / Equations                                                                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **N-gram Models**                  | Predict next word based on previous N−1 words                            | Simple, fast, good baseline, flexible N (tri-, penta-, etc.)            | Ignore syntax/semantics, fixed context window                      | Probability: $P(w_n \mid w_{n-N+1}, \dots, w_{n-1})$                                                                                                                                 |
| **Bayesian Techniques**            | Use Bayes’ theorem to calculate probability of hypothesis given evidence | Probabilistic reasoning, strong classification baseline                 | Context-insensitive, not generative, fails for ambiguous words     | $P(\text{hypothesis} \mid \text{evidence}) = \frac{P(\text{evidence} \mid \text{hypothesis}) P(\text{hypothesis})}{P(\text{evidence})}$                                              |
| **Markov Chains (HMMs)**           | Next state depends only on current state                                 | Intuitive, fast, used in PoS tagging, NER, text generation              | No long-term context, semantically limited                         | $P(X_{t+1} \mid X_1, \dots, X_t) = P(X_{t+1} \mid X_t)$                                                                                                                              |
| **Continuous Bag-of-Words (CBoW)** | Predict missing word from surrounding context                            | Learns dense word embeddings, captures local semantic similarity        | Context window limited, not suitable for long sequences/generation | Context vectors → hidden layer → output probabilities                                                                                                                                |
| **Embeddings**                     | Continuous vector representation of words                                | Encode semantics, syntactic relationships, reusable in downstream tasks | Context-independent, ignore pragmatics                             | Word vectors allow analogies, e.g., $\text{king}-\text{man}+\text{woman}\approx \text{queen}$                                                                                        |
| **MLPs**                           | Stack of neurons detecting features hierarchically                       | Flexible, foundation of neural networks, supports various tasks         | Requires careful design for large sequences                        | Forward pass: linear → activation → output; forms basis of deep networks                                                                                                             |
| **RNNs**                           | Maintain internal state for sequences                                    | Handle variable-length input, capture sequential dependencies           | Vanishing/exploding gradients, biased toward short sequences       | State update: $h_t = f(h_{t-1}, x_t)$                                                                                                                                                |
| **LSTMs**                          | RNNs with memory cells and gates                                         | Capture long-term dependencies, bidirectional, less biased by length    | Computationally expensive                                          | Use input, forget, output gates to control memory                                                                                                                                    |
| **Attention**                      | Weight input elements by relevance to query                              | Handles long-range dependencies efficiently, core of Transformers       | Quadratic complexity in sequence length                            | Equations: <br> $Q = X W_Q, K = X W_K, V = X W_V$ <br> $\text{scores} = \frac{QK^\top}{\sqrt{d_k}}$ <br> $\alpha = \text{softmax(scores)}$ <br> $\text{Attention}(Q,K,V) = \alpha V$ |
