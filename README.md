# 100+ LLM Interview Questions for Top Companies

This repository contains over 100+ interview questions for Large Language Models (LLM) used by top companies like Google, NVIDIA, Meta, Microsoft, and Fortune 500 companies. Explore questions curated with insights from real-world scenarios, organized into 15 categories to facilitate learning and preparation.

---

#### You're not alone—many learners have been reaching out for detailed explanations and resources to level up their prep.

#### You can find answers here, visit [Mastering LLM](https://www.masteringllm.com/course/llm-interview-questions-and-answers?previouspage=allcourses&isenrolled=no#/home).
#### Use the code `LLM50` at checkout to get **50% off**

---

![Image Description](interviewprep.jpg)

---
## Table of Contents

1. [Prompt Engineering & Basics of LLM](#prompt-engineering--basics-of-llm)
2. [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
3. [Document digitization & Chunking](#document-digitization-&-chunking)
4. [Embedding Models](#embedding-models)
5. [Internal Working of Vector Databases](#internal-working-of-vector-databases)
6. [Advanced Search Algorithms](#advanced-search-algorithms)
7. [Language Models Internal Working](#language-models-internal-working)
8. [Supervised Fine-Tuning of LLM](#supervised-fine-tuning-of-llm)
9. [Preference Alignment (RLHF/DPO)](#preference-alignment-rlhfdpo)
10. [Evaluation of LLM System](#evaluation-of-llm-system)
11. [Hallucination Control Techniques](#hallucination-control-techniques)
12. [Deployment of LLM](#deployment-of-llm)
13. [Agent-Based System](#agent-based-system)
14. [Prompt Hacking](#prompt-hacking)
15. [Miscellaneous](#miscellaneous)
16. [Case Studies](#case-studies)

---

## Prompt Engineering & Basics of LLM

- **What is the difference between Predictive/Discriminative AI and Generative AI?**


  Predictive/Discriminative AI: predict labels or outcomes given some input data. It models boundaries between classes (discriminates one thing from another). It learns the probability of label y given features x. It does not neccessarily learns how the input data was generated. the only objective is to predict outcomes correctly. <br>
Example: classification, regression, email spam or ham etc.

  Generative AI: Generate new data samples that look like the training data itself. It models how the data is distributed. It learns the fulls distribution p(x) or the joint distribution p(x, y). It can generate new x's, optionally conditioned on y (like text, images). <br>
Example: Writting emails. 
- **What is LLM, and how are LLMs trained?**

  LLM is a neural network based on Transformer architecture and trained on very large amount of text data. It predict sthe next token, word, subword.. given a sequence of previous tokens. LLM has billions or even trillions of parameters and works primarily on natural languages.

  Training LLMs: <br>
  Pretraining: Self supervised learning, to predict the next token. Objective is to learn general patterns in the data (language). Usually cross entropy loss function is used for optimization. <br>
  Fine-tuning: Supervised fine tuned on human annotated data or labeled data examples. Also reinforcement learning from human feedback (RLHF) where human preference guide the model outputs. Human preferences or ranking collected to train a "reward model" and LLM tries to maximize reward. 
- **What is a token in the language model?**

  A token is the smallest unit of text that the language model processes as input or output. It can be a word, subword, charachter, symbol, code etc. <br>
  Subword tokenization: common words like "cat" stays whole. Uncommon words like "unbelievableness" are broken into "un", "believable", "ness". This way model doesn't need to know every word and can handle rare, new or invented words.
  Tokenization: Splits text into tokens. Each token is mapped to a unique integer id.
  - Byte Pair Encoding (GPT-2, GPT-3)
  - WordPiece (BERT)
  - Unigram Language Model (SentencePiece)
  - Tiktoken (GPT-4)
- **How to estimate the cost of running SaaS-based and Open Source LLM models?**

    SaaS Based- Factors:
  - Model type (GPT-3.5, GTP-4)
  - Input tokes (promt size)
  - Output token (model response)
  - Frequenncy (how many request per hour/day)
  - Features used (any tools, function calling ect.)
  - total cost = (# of requests * avg tokens per request * price per 1K tokens)
    
  Open_source LLM- Factors:
  - Hardware (GPU, RAM, storage)
  - Cloud costs (AWS, Azure....)
  - Fine-tuning (training infrastructure cost)
    
  For open source, conside GPU hardware, inference throughput, hosting time, and operation complexity. 
  
- **Explain the Temperature parameter and how to set it.**

  In an LLM, after predicting the next token probability distribution, the temperature controls how much randomness we allow when sampling from that distribution.
  
  - T = 1.0, Default softmax behavior (normal distribution)
  - T < 1.0, Sharper, more confident, deterministic outputs. Model strongly picks high-probaility tokens.
  - T > 1.0, Flatter, more random, creative, surprising outputs. Lower and higher probaility tokens are equally likely.
  - T = 0, Greedy decoding - always pick the highest probability token, no randomness.
- **What are different decoding strategies for picking output tokens?**

  Once the LLM computes the probability diistribution over possible next tokens, decoding is the method by which the next token is chosen.

  - Greedy Decoding: No randomness, always choose the token with highest probability at each step.
  - Beam Search: Instead of just picking the top token, keep the Top N sequences at each step (beam size = N). Explore multiple promising options simultaneously. At the end pick the overall best sequence. More balanced and globally optimal outputs than greedy. Good for tanslation.
  - Sampling: Sample the next token according to the probability distribution. This introduces randomness and can generate creative, diverse responses. Good for story writting, brainstorming.
  - Top-k Sampling: Limit sampling to the top-k most likely tokens. Controlled randomness, still allowing diversity among strong options. Depens on the choice of K. K too large: lots of noise, K too small: deterministic. Like bias variance trade off.
  - Top-p Sampling (Nucleus Sampling): Dynamic version of top-k. Instead of a fixed k, choose the smallest set of tokens whose total probability mass exceeds a threshold p. This adaptively selects token set size based on context. Generally produces higher quality outputs that fixed top-k.
  - Temperature adjustment (see above)
- **What are different ways you can define stopping criteria in large language model?**

  Stopping Criteria determines when an LLM should stop generating tokes during inference.

  - Max token limit: Stop generation after a fixed number of tokens. Upper bound on cost, latency.
  - End-of-Sequence Token: LLMs are trained to emit a special [EOS] token when they think the output should stop. Automatically triggered in models trained with EOS tokens.
  - Stop Sequence: You define specific text strings - if any appear, generation stops. This is good for structured conversations, prompt templates.
  - Custom Logics or callbacks (Advanced): Implement custom logic to monitor generation in real time and halt based on lgic. Example:
      - Token patterns: Stop after 3 code blocks generation.
      - Legacy budget: stop if generation exceeds X ms.
  - Log Probability Threshold: Stop generation when the probability of all remaining token options becomes too flat or too sharp, indication uncertainity. This is rarely used. 
- **How to use stop sequences in LLMs?**

  A stop sequence is a specific string or token pattern that, when generated by the model, causes generation to stop immediately. Useful in dialogue agents, structured output formats, and ensuring clean output boundaries. Use stop sequence in combination of max tokens as a safety net.
  - OenAI API:
    ```

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What's your name?"},
        ],
        stop=["User:", "Assistant:"],  # custom stop strings
        max_tokens=100
    )
    print(response['choices'][0]['message']['content'])

    ```
  - Huggingface Transformers:
    ```
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2")
    
    output = generator(
        "Q: What is the capital of France?\nA:",
        max_new_tokens=50,
        eos_token_id=50256,  # EOS token
        pad_token_id=50256,
        stopping_criteria=[
            lambda input_ids, scores: any(
                tokenizer.decode(input_ids[0]).endswith(stop) for stop in ["\nQ:", "\nA:"]
            )
        ]
    )
    ```
  - Lnagchain or RAG Frameworks:
    ```
    llm = OpenAI(model="gpt-4", stop=["###", "\nUser:"])
    ```
- **Explain the basic structure prompt engineering.**

  
- **Explain in-context learning**
- **Explain type of prompt engineering**
- **What are some of the aspect to keep in mind while using few-shots prompting?**
- **What are certain strategies to write good prompt?**
- **What is hallucination, and how can it be controlled using prompt engineering?**
- **How to improve the reasoning ability of LLM through prompt engineering?**
- **How to improve LLM reasoning if your COT prompt fails?**

[Back to Top](#table-of-contents)

---

## Retrieval Augmented Generation (RAG)

- **how to increase accuracy, and reliability & make answers verifiable in LLM**
- **How does RAG work?**
- **What are some benefits of using the RAG system?**
- **When should I use Fine-tuning instead of RAG?**
- **What are the architecture patterns for customizing LLM with proprietary data?**

[Back to Top](#table-of-contents)

---

## Document digitization & Chunking 

- **What is chunking, and why do we chunk our data?**
- **What factors influence chunk size?**
- **What are the different types of chunking methods?**
- **How to find the ideal chunk size?**
- **What is the best method to digitize and chunk complex documents like annual reports?**
- **How to handle tables during chunking?**
- **How do you handle very large table for better retrieval?**
- **How to handle list item during chunking?**
- **How do you build production grade document processing and indexing pipeline?**
- **How to handle graphs & charts in RAG**

[Back to Top](#table-of-contents)

---

## Embedding Models

- **What are vector embeddings, and what is an embedding model?**
- **How is an embedding model used in the context of LLM applications?**
- **What is the difference between embedding short and long content?**
- **How to benchmark embedding models on your data?**
- **Suppose you are working with an open AI embedding model, after benchmarking accuracy is coming low, how would you further improve the accuracy of embedding the search model?**
- **Walk me through steps of improving sentence transformer model used for embedding?**

[Back to Top](#table-of-contents)

---

## Internal Working of Vector Databases

- **What is a vector database?**
- **How does a vector database differ from traditional databases?**
- **How does a vector database work?**
- **Explain difference between vector index, vector DB & vector plugins?**
- **You are working on a project that involves a small dataset of customer reviews. Your task is to find similar reviews in the dataset. The priority is to achieve perfect accuracy in finding the most similar reviews, and the speed of the search is not a primary concern. Which search strategy would you choose and why?**
- **Explain vector search strategies like clustering and Locality-Sensitive Hashing.**
- **How does clustering reduce search space? When does it fail and how can we mitigate these failures?**
- **Explain Random projection index?**
- **Explain Locality-sensitive hashing (LHS) indexing method?**
- **Explain product quantization (PQ) indexing method?**
- **Compare different Vector index and given a scenario, which vector index you would use for a project?**
- **How would you decide ideal search similarity metrics for the use case?**
- **Explain different types and challenges associated with filtering in vector DB?**
- **How to decide the best vector database for your needs?**

[Back to Top](#table-of-contents)

---

## Advanced Search Algorithms

- **What are architecture patterns for information retrieval & semantic search?**
- **Why it’s important to have very good search**
- **How can you achieve efficient and accurate search results in large-scale datasets?**
- **Consider a scenario where a client has already built a RAG-based system that is not giving accurate results, upon investigation you find out that the retrieval system is not accurate, what steps you will take to improve it?**
- **Explain the keyword-based retrieval method**
- **How to fine-tune re-ranking models?**
- **Explain most common metric used in information retrieval and when it fails?**
- **If you were to create an algorithm for a Quora-like question-answering system, with the objective of ensuring users find the most pertinent answers as quickly as possible, which evaluation metric would you choose to assess the effectiveness of your system?**
- **I have a recommendation system, which metric should I use to evaluate the system?**
- **Compare different information retrieval metrics and which one to use when?**
- **How does hybrid search works?**
- **If you have search results from multiple methods, how would you merge and homogenize the rankings into a single result set?**
- **How to handle multi-hop/multifaceted queries?**
- **What are different techniques to be used to improved retrieval?**
  

[Back to Top](#table-of-contents)

---

## Language Models Internal Working

- **Can you provide a detailed explanation of the concept of self-attention?**
- **Explain the disadvantages of the self-attention mechanism and how can you overcome it.**
- **What is positional encoding?**
- **Explain Transformer architecture in detail.**
- **What are some of the advantages of using a transformer instead of LSTM?**
- **What is the difference between local attention and global attention?**
- **What makes transformers heavy on computation and memory, and how can we address this?**
- **How can you increase the context length of an LLM?**
- **If I have a vocabulary of 100K words/tokens, how can I optimize transformer architecture?**
- **A large vocabulary can cause computation issues and a small vocabulary can cause OOV issues, what approach you would use to find the best balance of vocabulary?**
- **Explain different types of LLM architecture and which type of architecture is best for which task?**


[Back to Top](#table-of-contents)

---

## Supervised Fine-Tuning of LLM

- **What is fine-tuning, and why is it needed?**
- **Which scenario do we need to fine-tune LLM?**
- **How to make the decision of fine-tuning?**
- **How do you improve the model to answer only if there is sufficient context for doing so?**
- **How to create fine-tuning datasets for Q&A?**
- **How to set hyperparameters for fine-tuning?**
- **How to estimate infrastructure requirements for fine-tuning LLM?**
- **How do you fine-tune LLM on consumer hardware?**
- **What are the different categories of the PEFT method?**
- **What is catastrophic forgetting in LLMs?**
- **What are different re-parameterized methods for fine-tuning?**

[Back to Top](#table-of-contents)

---

## Preference Alignment (RLHF/DPO)

- **At which stage you will decide to go for the Preference alignment type of method rather than SFT?**
- **What is RLHF, and how is it used?**
- **What is the reward hacking issue in RLHF?**
- **Explain different preference alignment methods.**

[Back to Top](#table-of-contents)

---

## Evaluation of LLM System

- **How do you evaluate the best LLM model for your use case?**
- **How to evaluate RAG-based systems?**
- **What are different metrics for evaluating LLMs?**
- **Explain the Chain of Verification.**

[Back to Top](#table-of-contents)

---

## Hallucination Control Techniques

- **What are different forms of hallucinations?**
- **How to control hallucinations at various levels?**

[Back to Top](#table-of-contents)

---

## Deployment of LLM

- **Why does quantization not decrease the accuracy of LLM?**
- **What are the techniques by which you can optimize the inference of LLM for higher throughput?**
- **How to accelerate response time of model without attention approximation like group query attention?**

[Back to Top](#table-of-contents)

---

## Agent-Based System

- **Explain the basic concepts of an agent and the types of strategies available to implement agents**
- **Why do we need agents and what are some common strategies to implement agents?**
- **Explain ReAct prompting with a code example and its advantages**
- **Explain Plan and Execute prompting strategy**
- **Explain OpenAI functions strategy with code examples**
- **Explain the difference between OpenAI functions vs LangChain Agents**

[Back to Top](#table-of-contents)

---

## Prompt Hacking

- **What is prompt hacking and why should we bother about it?**
- **What are the different types of prompt hacking?**
- **What are the different defense tactics from prompt hacking?**

[Back to Top](#table-of-contents)

---

## Miscellaneous

- **How to optimize cost of overall LLM System?**
- **What are mixture of expert models (MoE)?**
- **How to build production grade RAG system, explain each component in detail ?**
- **What is FP8 variable and what are its advantages of it**
- **How to train LLM with low precision training without compromising on accuracy ?**
- **How to calculate size of KV cache**
- **Explain dimension of each layer in multi headed transformation attention block**
- **How do you make sure that attention layer focuses on the right part of the input?**


[Back to Top](#table-of-contents)

---

## Case Studies

- **Case Study 1**: LLM Chat Assistant with dynamic context based on query
- **Case Study 2**: Prompting Techniques

[Back to Top](#table-of-contents)

---

*For answers for those questions please, visit [Mastering LLM](https://www.masteringllm.com/course/llm-interview-questions-and-answers?previouspage=allcourses&isenrolled=no#/home).*

