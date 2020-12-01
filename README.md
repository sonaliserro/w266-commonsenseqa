# Making Sense of Commonsense
### How can language models perform better at commonsense reasoning?

Sonali Serro, Haerang Lee

*{sonaliserro,haeranglee}@berkeley.edu}*

##### *Abstract*

Commonsense reasoning has long been acknowledged as an essential component of natural language understanding. While humans have an innate ability to acquire and develop commonsense reasoning skills, it is still a challenge for machines. [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) is a benchmark dataset for commonsense question answering that continues to be difficult even for large-scale pretrained language models. 

Our research investigates two methods to improve a language model's performance on this task, namely, **intermediate task transfer learning** and **generative data augmentation**. We find positive results from the transfer learning approach - an improvement of 2.3% in CommonsenseQA development accuracy from our baseline, and also improved performance in especially low-resource settings. Our detailed error analysis reveals some of the commonsense reasoning skills acquired as a result of the intermediate task fine-tuning. We also observe encouraging results from the generative data augmentation approach, and we find that the language model tends to converge faster, potentially speeding up training by about 2 times.  

