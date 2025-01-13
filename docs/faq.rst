Frequently Asked Questions
==========================

This section addresses common questions about GCA Analyzer's metrics and functionality.

Participation Values
---------------------

Q: Why are some participation values negative?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Participation values are adjusted based on group size and represent deviation from perfectly equal participation. 
Negative values indicate contributions below the equal-participation amount, while positive values indicate 
contributions above it. A value of 0 means all participants contributed equally. This measurement allows us to 
intuitively see each participant's performance relative to equal participation.

For example, if there are 4 participants, the equal participation rate would be 0.25 (or 25%). If a participant 
contributes more than this, they will have a positive participation value, and if they contribute less, they 
will have a negative value.

Window Size Configuration
-------------------------

Q: What's the optimal window size?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The analyzer automatically finds the optimal window size based on the ``best-window-indices`` parameter. 
Lower values (e.g., 0.03) result in smaller windows, which may be more suitable for sparse conversations.

The window size affects how the analyzer measures interaction patterns:  
- Smaller windows: Better for detecting short-term interaction patterns  
- Larger windows: Better for capturing broader conversation dynamics  

Language Support
-------------------

Q: How to handle different languages?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The analyzer uses LLM (Large Language Model) models for text processing and supports multiple languages by default. 
For Chinese text, it uses the Chinese base model. The multilingual support includes:

- Automatic language detection  
- Language-specific text processing  
- Cross-language semantic analysis  

The default model used is ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``, which supports 
over 50 languages.
