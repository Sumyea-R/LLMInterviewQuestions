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

  Prompt engineering the practice of designing the input text given to a large language model to guide the output towards a desired format.
  - Instruction: Tell the model what you want it to do, explicitly and clearly.
  - Context: Give background information needed to complete the task better and reducing ambiguity.
  - Input Data: Give LLM some actual data that it needs to work on. This could be text, tables, code.
  - Output Format: Tell LLM how you want the answer structured. This will make parsing easier, more usability. 
    
- **Explain in-context learning**

  In context learning is when a large language model learns a task from examples provided directly in the input prompt, without any parameter updates (no gradient descent,no retraining). The model is shown some examples of input-output pairs inside the prompt. The model infers the task pattern just from reading examples. It continues the patters appropriately when given a new input. 
- **Explain type of prompt engineering**

  - Zero-Shot Prompting: Ask the model with no example to do a task. Only clear instructions.
  - One-Shot Prompting: Give one example on how to perform the task before asking the real query.
  - Few-Shot Prompting: Give multiple (2-5) examples in the prompt before querying. For domain specific tasks.
  - Chain-of-Thought (COT) Prompting: Encourage the model to reason step by step before giving the final answer. For mathmatical, logical tasks.
  - ReAct Prompting: Prompt the model to interleave resoning with tool use, like search, math, or API calls. Example:
    ```
    Question: What is the population of Berlin divided by the number of districts?

    Thought: I need to search for the population and district count.
    Action: Search["population of Berlin"]
    Observation: 3.6 million
    ...
    Final Answer: 3.6 million / 12 = 300,000

    ```
  - Instruction Prompting: Provide clear instructions in the prompt.
    ```
    You are a helpful assistant. Convert the list into JSON format with fields name and age.
    Input: John, 30; Alice, 25

    ```
  - Role-based Prompting: Assign a specific persona, role, or identity to the model. Best for chatbot assistants.
    ```
    You are a polite customer support assistant. Answer the following complaint:

    "My product arrived broken."

    ```
  - Multi-turn Prompting: For dialogue context. Include prior exchanges to preserve context across turns. For conversational agents.
    ```
    User: What are the symptoms of flu?  
    Assistant: Common symptoms include fever, cough, and sore throat.  
    User: And how can I treat it at home?

    ```
  - Soft Prompting/Prompt Tuning: (Advanced) Instead of textual prompts, inject learned embeddings into the input to guide the model. 
- **What are some of the aspect to keep in mind while using few-shots prompting?**

    - Example Quality: correct representative, well formatted
    - Consistent Formatting
    - Ordering of example can matter: oreder example from simple to complex. LLM processes tokens sequentially. So put easier clearer examples first to establish the pattern.
    - Use a reasonable number of examples. If too high, waste of context space, confused model.
    - Task relevant diversity: Include diverse example to include edge cases, typical outputs, different categories, etc.
    - Clear task definition and instructions.
    - Watch for overfitting for examples.
    - Try different example sets for accuracy, style, faithfulness. 
- **What are certain strategies to write good prompt?**

  Using a clear format that repeats the pattern you want the model to follow. Give the model a role by starting the prompt telling the model who it is. When needed use few shot examples. Specify the output format and avoid prompt overload. Stay within token limits and stay focused. Iteratively test and refine prompts. Clean data before giving it to the model, and parse/validate the result after. Always good to use stop sequence or max tokens to constrain oputputs. 
- **What is hallucination, and how can it be controlled using prompt engineering?**

  Hallucination is when an LLM generates factually incorrect, fabricated, or misleading content, even though it appears confident. It happens when:
   - the model is trained to produce plausible continuations, not verified facts.
   - Lack of grounding, when the model doesn't access external knowledge.
   - When model blends similar training data to produce likely sounding but incorrect outputs.

  Controlling Hallucination:
    - Chain-of-thought prompting: Step by step reasoning
    - Use role assignment to limit speculation.
    - Provide context and source materials.
    - Use RAG
    - Use constrains like, "Only answer based on provided information. Do not make assummptions"
    - Ask for source where the model got the information.
- **How to improve the reasoning ability of LLM through prompt engineering?**

    - COT Prompting: It slows down token prediction, encouraging intermediate logical states.
    - Self-Ask Prompting: Decompose the complext question into simpler sub-queries and solve them one at a time.
      ```
      Q: Who was the president of the U.S. when the Apollo 11 mission landed?

      A: Let's break it into subquestions:
      - When did Apollo 11 land? → 1969
      - Who was president in 1969? → Richard Nixon
      
      Answer: Richard Nixon

      ```
    - Provide reasoning examples: Few-Shot CoT.
    - Constrain output structure for reasoning.
      ```
      First, list the relevant facts.  
      Then, explain the logical steps.  
      Finally, provide the answer.

      ```
    - Role-assignment
- **How to improve LLM reasoning if your COT prompt fails?**

  Why CoT fails:
    - Model too small
    - Prompt lacks good reasoning examples
    - Prompt too vague
    - Task is fundamentally hard (might need external tool)

  If CoT fails, switch to strategies like task decomposition, reasoning formats, few-shot examples, role assigning, RAG or even tool-augmented prompting.
    - Self ask or decompositioning prompting.
    - Use structured prompt templates
    - Few-shot CoT
    - Use ReAct prompting
    - Assign domain specific role
    - Use iterative prompting: ask, check, revise
    - Try rephrasing the question
    - Fallback to retrieval or Tool use (RAG, Agents)

[Back to Top](#table-of-contents)

---

## Retrieval Augmented Generation (RAG)

- **how to increase accuracy, and reliability & make answers verifiable in LLM**

  - Ground the model using external Context (RAG). Fetch documents from trusted sources, database, search engines, pdfs... This makes answers verifiable via checking it against the context.
  - Chain-of-thought
  - Assigning role
  - Use post-answer verification prompt. Ask the model to critique its own answer or provide evidence, like sources. This is useful in multi-agent system.
  - Use multiple LLMs or multiple passes and compare results. Use voting, ensemble to pick answers.
  - Use temperature + decoding constraints: Set low temperature to limit randomness; improving accuracy and reliability.
- **How does RAG work?**

  RAG is a framework where an LLM generates responses based on retrieved external documents, rather than relying only on its external knowledge. It helps prevent hallucinations. Following shows how RAG works, step-by-step.
    - Query Embedding: When user asks a question, the query is converted into a vector, using a embedding model (sentence transformer, OpenAI embeddings)
    - Document Retrieval: The query embedding is used to perform vector similarity search in a document database. (using FAISS, Pinecone, Qdrant, Elastisearch) to retirieve top-k most relevant documents/passages.
    - Context Injection: The retrieved documents are added to the prompt given to the LLM.
    - Answer Generation: The LLM reads the combined question + retrieved context and generates an answer based on that. 
- **What are some benefits of using the RAG system?**

  RAG systems improve LLM accuracy and trustworthiness by grounding generation in real documents, reducing hallucination, supporting pricate or domain-specific knowledge and enabling easy updates without retraining the model. 
  - Improves accuracy and reduces hallucinations.
  - RAG enables traceability, Makes responses verifiable
  - RAG adds knowledge without retraining the LLM.
  - Adds domain specific private knoowledge
  - Improves performance on Long-Tail queries.
- **When should I use Fine-tuning instead of RAG?**

    Fine-tuning is better when you need the model to learn patterns, tasks, or skills instead of just recalling information/facts. Like sentiment classification, style imitation (write like Shakespeare). If you don't have external documents or retrieval is weak. RAG introduces retrieval latency and more computation per query. If you need low-latency or offline generation then fine-tuning is better. 
- **What are the architecture patterns for customizing LLM with proprietary data?**
  To customize an LLM with proprietary data, choose between RAG, fine-tuning, prompt-based retrieval or agent based architectures depending on whether the task needs dynamic grounding, behavior learning, structured reasoning, or too orchestration.
  - RAG
  - Fine-Tune if possible (expensive and slower to deploy)
  - Agent Based LLM + Tool Calling. LLM orchestrate (LangChain, OpenAI) tools that query proprietary data. 

[Back to Top](#table-of-contents)

---

## Document digitization & Chunking 

- **What is chunking, and why do we chunk our data?**

  Chunking is the process of splitting largedocuments or unstructured text into smaller pieces (chunks) to make them suitable for embedding, storage, and retrieval in an LLM based system. Each chunk becomes a unit of retrieval, when a user submits a query, relevant chunks are retrieved via vector similarity search and injected into prompt.

  Why Chunking: LLMs can't take large documents, chunking ensures pieces fit within the context window size. This way LLM retrieves parts, not entire documents.
  Chunking ensures efficient embeddings for models, and embedding models work best on small, coherent segments. Better chunking leads to better matches during similarity search.
- **What factors influence chunk size?**

  Chunk size refers to the number of tokens, words or characters in each piece of text you split from a larger document to store and retrieve during RAG or embedding workflows. Chunk size is influenced by different factors.
    - LLM context window size. You can't inject more context than the model can handle. Chunk size must be small enough so multiple chunks can fit with the user query and prompt instructions.
    - Embedding model limits: Embedding models have a maximum token input size, usually 512-8192 tokens.
    - Semantic Coherence: chunks should contain complete thoughts, not fragments.
    - Document Type and structure: for HTML/Markdown file headers, sections need to be considered. PDFs consider paragraphs , line breaks. For code preserve full functions or classes. For tables avoid breaking across rows/columns.
    - Use Case requirements: use case FAQ, chunk size short 100-300 tokens. For legal/technical docs, 300-800 tokens.... etc.
 
  The goal is to balance precison, context and performance.
- **What are the different types of chunking methods?**

    - Fixed-size Chunking: Simple sliding window. Split by tokens, (e.g., every 500 tokens)
    - Semantic or sentence based chunking: Split by paragraphs, sentences, or sections. This preserves meaning boundaries. This produces more coherent chunks.
    - Metadata aware chunking: Preserve document structure, like heading, bullet points, tables, footnotes. This requires parsing logic.
    - Hybrid Chunking: Recursive splitting. Combine semantic and fixed-length. First try to split paragraphs. If too large, split again by sentence. Then finally by characters. 
- **How to find the ideal chunk size?**

    To find the ideal chunk size, start with 200-800 token ranges, use semantic chunking with overlap, and empirically evaluate retrieval hit rate and LLM answer quality to choose the best trade-off between context, cost, and accuracy.
    - Know the limit: What is the model's max context length? How much of that needs to be reserved for? Keep each chunk well below 1/4 to 1/3 of the context window to allow for multiple chunks per prompt.
    - Start with smart defaults: Short factual documents: 200-300 tokens. Paragraph-level QA: 400-600 tokens. Code or structured text: 600-1000 tokens. Include 20-30% overlap to preserve continuity.
    - Implement Recursive Chunking: Use chunker that split on semantic boundaries (paragraphs, sentences). Fall back to token count if boundaries are too large. Use tools like LangChain: RecursiveCharacterTextSplitter, Hugging Face: sentence-transformers or nltk sentence tokenizers.
    - Empirically evaluate chunk size: Once you have indexed chunks of different sizes (200, 400, 600, 800 tokens) evaluate using
        - Hit rate: Does the correct chunk get retrieved?
        - Answer accuracy: Does the LLM produce correct, grounded answers?
        - Latency/Cost: larger chunks = fewer retrieval calls, but higher token cost.
        - Hallucination rate: are answers using irrelevant context?
    <br>this evaluations can be done using LangChain eval, trulens, Regas (Retrieval-Augmented Generation Evaluation)
    - Compare retrieval performances across chunk sizes
        - Precision: Is the retrieved content more focused?
        - Recall: does the chunk contain all needed information?
        - answer quality: Does LLM improve or degrade?
  <br> too small chunks - higher precision, but may miss context. Too large chunks - recall improves, but precission drops, more hallucinations.
    - Choose the sweet spot: Select the chunk size that offers high retrieval accuracy, answer groundedness and acceptable latency and cost.
- **What is the best method to digitize and chunk complex documents like annual reports?**

  The best way to digitize and chunk complex documents like annual reports is to extract structured content using layout-aware parsers, segment by section headers, chunk sematically with token limits, and store chunks with metadata for traceable retrieval in a RAG piepline.
    - Use structured PDF parser (PyMuPDF/ pdfplumber) to extract clean text + structure. This will preserve headings, table regions, figure captions, page breaks.
    - Segment the document into logival sections. Split based on headings, structural markers (line patterns, indentation). We want semantic chunk boundaries, no mid sentence or mid-table breaks.
    - Once the logical chunk sections are extracted, apply token-based or recursive chunking within each section. LangChain → RecursiveCharacterTextSplitter with chunk size (e.g., 600 tokens) and overlap.
    - Preserve metadata for retrieval. Add metadata for section title, page number, document type...
    - Store in a vector store. 
- **How to handle tables during chunking?**

    To handle tables during chunking, detect them using structure-aware parsers, keep them as atomic blocks or row-based chunks with headers preserved, format them cleanly, and attach metadata to support reliable retrieval and reasoning RAG pipelines.
    - Detect tables during parsing. pdfplumber, Unstructured..
    - Treat each table as an atomic chunk. If table fits within chunk limits (<600 tokens), keep it whole. If too large, chunk by row group, don't split mid row and always include the header row in each chunk.
    - Preserve table metadata. Table caption, section title, row range, page number..
    - Store table chunks as structured text or markdown (preferred for LLMs).
- **How do you handle very large table for better retrieval?**

    - Row-based chunking with header repetition.
    - Add metadata to each chunk
    - Decompose table into subtbles if table has hierarchy. ex: table for revenue by region, table for Q1-Q4 results.
- **How to handle list item during chunking?**

    To handle list items during chunking, detect and preserve their structure, chunk them by logical groups without breaking steps, format them in markdown, and attach metadata to support accurate retrieval.
    - Detect Lists during parsing. unstructures.
    - Keep list items in the same chunk if possible.
    - Chunk large lists by logical groups. ex. based on categories. ALways repeat the list title in each chunk to maintain context.
    - Preserve list formatting. Markdown style best for OpenAI.
    - Attach metadata to list chunks, list title, item start index, item end index, section title, page number, parent heading, etc.
    - Avoid splitting nested lists mid-depth. Chunk the whole sublist with its parent. 
- **How do you build production grade document processing and indexing pipeline?**\

    To build a production-grade document processing pipeline, parse layout-aware text, chunk semantically, enrich with metadata, embed each chunk, index in a vector store with hybrid search, and expose it through a RAG layer with monitoring and evaluation built in.

    - Parsing and Cleaning: normalize layout, remove noise, detect titles,paragraphs, lists, tables, images, charts
    - Semantic chunking
    - Metadata extraction and enrichment, attach useful info for filtering and reranking
    - Embedding: convert each chunk into vector representation for similarity search.
    - Indexing: Vector DB (FAISS< Qdrant, Weaviate, Pinecone)
    - Retrieval and Generation
- **How to handle graphs & charts in RAG**

    To handle charts and graphs in RAG, extract textual summaries describing their structure and key trends, embedd those summaries for retrieval and optionally link to the original image for UI display or vision-based models.
  
    - Extract or describe charts during ingestion. Use OCR chart parser (PaddleOCR/ Tesseract) to extract titles, axis labels, legends, data points and generate textual summary, like what the axes represents, key insights etc.
      ```
      Chart Title: Quarterly Revenue by Region  
      X-Axis: Quarter (Q1–Q4)  
      Y-Axis: Revenue in USD  
      Observation: Revenue increased steadily in Europe, with a 20% jump from Q2 to Q3.

      ```
    - Link the image with the textual description. (optional)
    - Embedd the textual summary.
    - Optional: Use multimodal models for visuals (GPT-4V, Gemini Pro Vision) works best for complex visuals (scatter plot, medical imaging)
    - Store charts as separate chunks with metadata.
   

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

