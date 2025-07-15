Advanced Usage
==============

This guide covers advanced features and configurations of GCA Analyzer.

Configuration Options
-----------------------

Language Model Configuration
-----------------------------

.. note::
   Ensure you have the appropriate language model credentials configured before using these features.

1. Using Custom LLM Models:

.. code-block:: python

   from gca_analyzer import LLMTextProcessor
   
   processor = LLMTextProcessor(
       model_name='your-model-name',
       mirror_url='your-model-mirror'
   )
   analyzer = GCAAnalyzer(llm_processor=processor)

2. Configuring Analysis Parameters:

.. code-block:: python

   from gca_analyzer import Config, WindowConfig, ModelConfig, LoggerConfig
   
   config = Config(
         window=WindowConfig(
            best_window_indices = 0.3 # Percentage of participants to consider for best window
            act_participant_indices = 2 # Number of contributions from participants considered as active participants
            min_window_size = 2
            max_window_size = None
         ),
         model=ModelConfig(
              model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
              mirror_url="https://modelscope.cn/models"
         ),
         logger=LoggerConfig(
              console_level='DEBUG',
              log_file='gca_analyzer.log'
         )
   )
   analyzer = GCAAnalyzer(config=config)

Advanced Analysis Features
---------------------------

Participation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Analyze participation patterns:

.. code-block:: python

   # Get detailed participation metrics
   metrics = analyzer.analyze_conversation('conv_id', data)
   participation = metrics[['Pa', 'Pa_average', 'Pa_std']]

Interaction Analysis
~~~~~~~~~~~~~~~~~~~~~

Analyze interaction patterns:

.. code-block:: python

   # Get interaction metrics
   interaction_metrics = metrics[[
       'internal_cohesion',
       'responsivity',
       'social_impact'
   ]]

Content Analysis
~~~~~~~~~~~~~~~~

Analyze content patterns:

.. code-block:: python

   # Get content metrics
   content_metrics = metrics[['newness', 'comm_density']]


Visualization
--------------

Create visualizations of analysis results:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot participation patterns
   plt.figure(figsize=(10, 6))
   metrics['Pa'].plot(kind='bar')
   plt.title('Participation Patterns')
   plt.show()

