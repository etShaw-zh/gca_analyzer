Advanced Usage
==============

This guide covers advanced features and configurations of GCA Analyzer.

Configuration Options
-----------------------

Window Size Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The analyzer automatically finds the optimal window size for analysis:

.. code-block:: python

   analyzer = GCAAnalyzer()
   
   # Custom window parameters
   window_size = analyzer.find_best_window_size(
       data,
       best_window_indices=0.3,
       min_num=2,
       max_num=10
   )

Advanced Analysis Features
---------------------------

Participation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Understanding participation values:

- Values are adjusted based on group size
- Negative values: contributions below equal-participation amount
- Positive values: contributions above equal-participation amount
- Zero: perfectly equal participation

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

   from gca_analyzer import Config
   
   config = Config(
       best_window_indices=0.2,
       min_window_size=2,
       max_window_size=8
   )
   analyzer = GCAAnalyzer(config=config)

Performance Considerations
--------------------------

* For large conversations (>1000 messages), consider batch processing
* Memory usage scales with conversation size and window parameters
* Use appropriate window sizes for optimal performance

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

