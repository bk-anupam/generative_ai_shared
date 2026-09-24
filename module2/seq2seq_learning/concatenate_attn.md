# Concatenate Attention in an LSTM Encoder–Decoder

An **attention mechanism** lets a neural network **assign different weights to available representations and combine them according to their relevance to the current task or step**.

The soft attention used here involves three operations:

1. **Score:** Compare a **query** with each available representation to calculate relevance scores.
2. **Normalize:** Apply softmax to turn the scores into attention weights.
3. **Combine:** Compute a weighted sum of the representations, producing a **context vector**.

$$
e_i=\operatorname{score}(q,h_i),
\qquad
\alpha_i=\frac{\exp(e_i)}{\sum_j\exp(e_j)},
\qquad
r=\sum_i\alpha_i h_i.
$$

Here, $q$ is the query, $h_i$ is the encoder output at source position $i$, $e_i$ is its relevance score, $\alpha_i$ is its attention weight, and $r$ is the context vector. The sums run over valid source positions. These equations describe one attention calculation; below, we add a time index $t$ when repeating it at each decoder step.

For example, when generating **“apples”** in a Hindi-to-English translation, attention can give a larger weight to the encoder representation of **“सेब.”** The context then contains a larger contribution from that position.

The query depends on the architecture: the translation model described here uses the **previous decoder hidden state**, while the [tweet classifier](disaster_tweets_with_attention.ipynb) uses the **encoder's final hidden representation**.

Let’s consider an **LSTM encoder–decoder for translation with concatenate attention**. We will separate three things throughout:

- **Encoder states:** representations of the source sentence.
- **Decoder state:** a representation of what the decoder has processed so far.
- **Attention context:** source information selected for the current decoding step.

Suppose we translate:

> **Source:** “das buch ist rot”  
> **Target:** “the book is red”

We will omit special tokens in the diagrams for clarity.

## Diagram: attention at each decoder step

![Concatenate attention: recurrent encoder LSTM cells produce encoder outputs for alignment scoring and a weighted sum; the resulting context feeds both the decoder LSTM and vocabulary projection.](assets/concatenate_attention_v2.png)

At step $t$, the previous hidden state $s_{t-1}$ queries the encoder outputs to produce scores, weights, and the context $r_t$. The orange path supplies this context to both the LSTM input and the vocabulary projection. The LSTM produces updated hidden and cell states $(s_t,m_t)$; the next step recomputes attention using $s_t$ and the same encoder outputs. The vocabulary softmax produces a probability distribution $p_t$, from which a token is selected or sampled.

*Each $h_i$ is a contextual encoder output; source-token inputs include embedding lookup, omitted for clarity. Encoder memory $H$ collects these outputs. *

## A Hindi-to-English example: attention and word order

Consider another translation:

> **Hindi:** मैं सेब खाता हूँ
>
> **English:** I eat apples.

Hindi places the verb after the object in this sentence, while English places it before the object. Attention lets the decoder consult the source positions useful for each English token, even when the word order changes.

For this teaching example, treat each displayed Hindi word as one token and omit punctuation and special tokens. A real tokenizer may split words into smaller units. At successive decoder steps, the attention weights might be:

| Generated English token | मैं | सेब | खाता | हूँ |
|---|---:|---:|---:|---:|
| **I** | **0.85** | 0.05 | 0.05 | 0.05 |
| **eat** | 0.05 | 0.05 | **0.65** | **0.25** |
| **apples** | 0.05 | **0.85** | 0.05 | 0.05 |

These are **illustrative weights, not measured model outputs**. Each row is a separate attention distribution over the source positions and sums to 1. The row labels identify the tokens being generated; those tokens are not supplied to the attention calculation as inputs.

When generating **“eat,”** the decoder attends mainly to **“खाता हूँ.”** When generating **“apples,”** it shifts back to **“सेब.”** Thus, attention need not move left to right or assign each output token to exactly one source token.

Let $h_1,h_2,h_3,h_4$ be the encoder outputs for मैं, सेब, खाता, and हूँ, respectively. For the third English token, the context is:

$$
r_3=0.05h_1+0.85h_2+0.05h_3+0.05h_4.
$$

The encoder outputs stay fixed throughout this translation. At each decoder step, the previous decoder hidden state supplies a new query, producing new alignment scores, attention weights, and a context vector. In the architecture shown above, $r_3$ helps update the decoder's hidden and cell states and also enters the vocabulary projection alongside the updated hidden state to predict “apples.”

The sections below return to the German-to-English example and explain these calculations in detail.

## 1. The encoder reads the source sentence

The encoder processes source tokens and produces a hidden state at every position:

$$
h_1,\ h_2,\ h_3,\ h_4.
$$

```text
 das       buch       ist        rot
  │          │         │          │
  ▼          ▼         ▼          ▼
Encoder → Encoder → Encoder → Encoder
  │          │         │          │
 h₁         h₂        h₃         h₄
```

Each $h_i$ is a **contextual representation**, not simply the embedding of word $i$.

For a unidirectional LSTM, $h_i$ depends on the source tokens up to position $i$. For a bidirectional LSTM, it combines information from both directions.

Without attention, the decoder receives the encoder’s final states and relies on them for the whole translation. **With attention, we also retain all the encoder outputs** so the decoder can consult them later.

## 2. Initialize the decoder

The decoder needs initial hidden and cell states. These can come from the encoder’s final states, possibly through a learned transformation if dimensions differ.

We will write:

- $s_{t-1}$: decoder hidden state before generating target token $t$.
- $m_{t-1}$: decoder LSTM cell state.
- $y_{t-1}$: previous target token.

We use $m$ for the decoder cell state to avoid confusing it with the attention context.

At the first step, the previous token is `<sos>`.

There are several valid ways to arrange attention and the decoder update. **Here, we compute attention using $s_{t-1}$, then use the resulting context in the decoder update.** This matches the previous-decoder-state explanation you were reading.

## 3. Compare the decoder state with every encoder state

At target step $t$, attention asks:

> Given the decoder’s current situation, which source representations would be useful for generating the next token?

The decoder state $s_{t-1}$ acts as a **query**. We compare it with each encoder state $h_i$.

For concatenate attention, we first concatenate the two vectors:

$$
z_{t,i}=[s_{t-1};h_i].
$$

The semicolon means concatenation along the feature dimension.

For example, if

$$
s_{t-1}=[0.2,\ 0.7],
\qquad
h_i=[0.5,\ -0.1,\ 0.4],
$$

then

$$
z_{t,i}=[0.2,\ 0.7,\ 0.5,\ -0.1,\ 0.4].
$$

**This is the concatenation that gives concatenate attention its name.** We concatenate the decoder state with an encoder state—not with an already-computed context vector.

The same decoder state is paired with every source position:

```text
[sₜ₋₁ ; h₁]   [sₜ₋₁ ; h₂]   [sₜ₋₁ ; h₃]   [sₜ₋₁ ; h₄]
```

## 4. Turn each pair into one scalar relevance score

Concatenation alone does not tell us how relevant a source position is. A small learned neural network computes that relevance:

$$
u_{t,i}=\tanh\left(W_a[s_{t-1};h_i]+b_a\right),
$$

followed by

$$
e_{t,i}=v_a^\top u_{t,i}.
$$

The quantities have different roles:

| Quantity | Meaning |
|---|---|
| $[s_{t-1};h_i]$ | Decoder and encoder features placed together |
| $u_{t,i}$ | Intermediate alignment feature vector |
| $e_{t,i}$ | One scalar compatibility score for source position $i$ |

$W_a$, $b_a$, and $v_a$ are learned parameters. **The same scoring network is used at every source position and every decoding step.**

The scoring network consists of two fully connected (FC) layers:

1. **FC layer followed by a tanh activation:** $W_a[s_{t-1};h_i]+b_a$ applies the first FC layer to the concatenated states. Applying $\tanh$ gives $u_{t,i}$, the **intermediate alignment feature vector**. Thus, $u_{t,i}$ is the output of this layer and activation, rather than the layer itself.
2. **FC projection to a scalar:** $v_a^\top u_{t,i}$ applies a second FC layer with one output and no bias in this formulation. It projects the intermediate alignment feature vector into the **scalar alignment score** $e_{t,i}$. No activation is applied at this stage; softmax subsequently normalizes the scores across source positions.

The first layer learns nonlinear compatibility features, and the second learns how to combine those features into one score.

A larger score means greater relative compatibility. Scores are not probabilities: they can be negative and need not sum to one.

## 5. Convert scores into attention weights

Suppose the decoder is about to generate “book.” It has one score for each source position:

$$
e_{t,1},e_{t,2},e_{t,3},e_{t,4}.
$$

Apply softmax **across source positions**, holding the target step $t$ fixed:

$$
\alpha_{t,i}
=
\frac{\exp(e_{t,i})}
{\sum_{k=1}^{T_x}\exp(e_{t,k})}.
$$

Here $T_x$ is the source length.

The resulting weights satisfy:

$$
\alpha_{t,i}\geq 0,
\qquad
\sum_i\alpha_{t,i}=1.
$$

An illustrative distribution might be:

| Source token | Attention weight |
|---|---:|
| das | 0.05 |
| buch | 0.80 |
| ist | 0.10 |
| rot | 0.05 |

These values are invented to explain the calculation, not measured model outputs.

This is **soft attention**: it distributes weight across source positions rather than selecting only one.

For padded batches, padding scores must be masked before softmax so padding receives no weight.

## 6. Construct the context vector

We now combine the encoder states using the attention weights:

$$
r_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i.
$$

For the illustrative distribution:

$$
r_t=
0.05h_1+0.80h_2+0.10h_3+0.05h_4.
$$

The result is **one context vector for the current target step**.

A small numerical example makes the operation concrete. Suppose there are three encoder states:

$$
h_1=[1,0],\qquad
h_2=[0,2],\qquad
h_3=[1,1],
$$

with weights

$$
\alpha_t=[0.1,0.7,0.2].
$$

Then

$$
\begin{aligned}
r_t
&=0.1[1,0]+0.7[0,2]+0.2[1,1]\\
&=[0.3,1.6].
\end{aligned}
$$

Notice that:

- The context has the **same feature dimension as each encoder state**.
- It is a weighted combination of representations, not a selected word.
- Concatenation was used to compute scores; the weighted sum constructs the context.

## 7. Use the context to update the decoder and predict a token

In the arrangement we are following, the decoder receives the previous token’s embedding together with the attention context:

$$
(s_t,m_t)
=
\operatorname{LSTM}_{\text{dec}}
\left(
[E_y(y_{t-1});r_t],
(s_{t-1},m_{t-1})
\right).
$$

Here $E_y(y_{t-1})$ is the embedding of the previous target token.

The decoder now combines:

- Its previous hidden and cell states.
- The previous target token.
- Source information retrieved through attention.

A vocabulary projection can then use both the updated decoder state and context:

$$
\ell_t=W_o[s_t;r_t]+b_o.
$$

The vector $\ell_t$ contains one logit per target-vocabulary token. A vocabulary softmax gives:

$$
P(y_t\mid y_{<t},X)=\operatorname{softmax}(\ell_t).
$$

**There are two different softmax operations here:**

| Softmax | Operates over | Answers |
|---|---|---|
| Attention softmax | Source positions | Which source representations should contribute? |
| Output softmax | Target vocabulary | Which token should be generated next? |

During inference, a decoding strategy such as greedy selection chooses the next token from the output distribution.

## 8. Repeat with a new attention distribution

After generating a token, the decoder state changes. At the next step, attention uses the new state to compute new scores and weights:

$$
s_t
\longrightarrow
e_{t+1,i}
\longrightarrow
\alpha_{t+1,i}
\longrightarrow
r_{t+1}.
$$

When generating “book,” attention might emphasize `buch`. Later, when generating “red,” it might emphasize `rot`.

**The encoder outputs remain the same during decoding of a given source sentence; the attention weights and context vector change at each target step.**

Attention does not have to move left to right. It can revisit positions or distribute weight over several positions, which is useful when source and target languages have different word orders.

Generation continues until `<eos>` or a length limit.

## 9. How does attention learn where to look?

Usually, we do not provide labels saying “look at `buch` when predicting `book`.”

Instead, the model is trained to predict the correct target tokens using a translation loss such as cross-entropy:

$$
\mathcal L=-\sum_t\log P(y_t^*\mid y_{<t}^*,X),
$$

where the stars denote reference tokens under teacher forcing.

The loss backpropagates through:

```text
Token prediction
      ↓
Decoder and context vector
      ↓
Attention weights
      ↓
Compatibility scores
      ↓
Scoring parameters and encoder representations
```

This jointly trains the encoder, decoder, and attention parameters. Attention learns weighting patterns that help the prediction task; it is not guaranteed to discover a clean one-to-one linguistic alignment.

Also, the decoder does **not** receive the current correct target token when computing its attention. During teacher forcing it receives previous reference tokens; the current reference token is used to calculate the loss.

## 10. Why this helps with the fixed-context bottleneck

Without attention, information about an early source token must survive through the encoder’s final-state summary and subsequent decoder updates until needed.

With attention, the decoder has an additional route:

$$
\text{encoder state at position }i
\rightarrow
\text{current attention context}
\rightarrow
\text{current prediction}.
$$

The context vector is still fixed-size, but it is **recomputed at every target step from a collection of encoder states whose size grows with the source length**. The model no longer needs to make one initial summary serve every output position equally well.

Finally, some architectures first update the decoder and then use $s_t$ to compute attention, instead of using $s_{t-1}$ as above. Both arrangements are valid. When reading equations or code, identify **which decoder state supplies the query and where the resulting context enters the prediction process**.
