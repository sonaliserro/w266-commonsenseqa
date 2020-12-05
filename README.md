# Making Sense of Commonsense
### How can language models perform better at commonsense reasoning?

Sonali Serro, Haerang Lee

*{sonaliserro,haeranglee}@berkeley.edu*

##### *Abstract*

Commonsense reasoning has long been acknowledged as an essential component of natural language understanding. While humans have an innate ability to acquire and develop commonsense reasoning skills, it is still a challenge for machines. [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) is a benchmark dataset for commonsense question answering that continues to be difficult even for large-scale pretrained language models. 

Our research investigates two methods to improve a language model's performance on CommonsenseQA, **intermediate task transfer learning** and **generative data augmentation**. We find positive results from the transfer learning approach as development accuracy increases by 2.3% from the baseline. Additionally, we observe improved model performance in especially low-resource settings. Our error analysis reveals some of the commonsense reasoning skills acquired from the intermediate task fine-tuning. The generative data augmentation method also shows encouraging results as we find that the language model converges faster, potentially reducing training time by half.


