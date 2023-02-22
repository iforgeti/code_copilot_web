# code_copilot_web

In this work, I use python code dataset from huggingface course https://huggingface.co/course/chapter7/6?fw=pt which consist of "pandas", "sklearn", "matplotlib", 

"seaborn" .

Since the dataset is too large, I split a little data from all dataset.In python code, There are space and \n too, So we have 2 option. 

First, we may write code to add \n and space to result if they find ':' but it still hard. Second, just tokenize \n and space which spacy seem can do that.

This time, I try use \n and space token to train model (just to see if it can learn about spacing and new line too).




In summary

 - i think using \n and space is not so good idea because vocab size will be a lot bigger.
 
 - we may clean data first to improve model.  
