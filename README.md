# ffr-v1
Towards developing a Robust Translation Model for African languages: Pilot Project FFR v1.0. 

"FFR v1.0" is a Fon-French translation model, trained on 117,089 parallel sentences using neural machine translation with attention.
While it could be observed that Masakhane https://www.masakhane.io/ , an online community of African researchers working on machine 
translation for African languages, have generated translation models and baselines from/to many African languages, however,
"Project FFR v1.0” is the first to make this effort on a large scale, by taking time to painstakingly amass a large 
training dataset and exploring techniques to work with the Fon diacritics for better translation accuracy
in order to achieve a publishable model which may be used by people to a certain degree of reliability.

Part of the research methodology used by the researchers in sourcing the data for this research includes rigorous compilation through 
“web-scraping,”* and  “parsing open source dataset websites. Through these efforts, we obtained 117,029 Fon-French parallel words and 
sentences, which we used for the pilot stage. Furthermore, the dataset was specially cleaned, pre-processed and tokenized, 
preserving the diacritics and special characters of the Fon alphabet. The owners of the website were contacted and permission was granted
to collect the data on their website.

FFR v1.0 was trained for 5 days, using the Paperspace cloud computation virtual machine and the code for the model was inspired from [1] 
and [2], with our added contributions to address the Fon diacritics.

[1] : Deep Learning for NLP, Jason Brownlee - Section 9 : Machine Translation
[2] : Tensorflow Tutorial on Neural Machine Translation with Attention Mechanism : 
      https://www.tensorflow.org/tutorials/text/nmt_with_attention

The project has been led so far by the edAI (https://twitter.com/edAIOfficial) researchers : 
Chris EMEZUE (https://twitter.com/ChrisEmezue)and Bonaventure DOSSOU (https://twitter.com/bonadossou)

Our work gave us an overall BLEU score of 44.38.
The model training and the bleu score distribution along the test dataset plots were provided too
Yyou can find all the results and summary about the are available in the repository FFR pdf file
