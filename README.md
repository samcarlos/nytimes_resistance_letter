# nytimes_resistance_letter
Which senior cabinet wrote nytimes 'resistance' article? 

Below ranking of each cabinent member. Azar, Acosta, or Haley are most likely to have written article based on data analysis. 


['rick perry', 1]
['Kelly_John', 2]
['kirstjen nielsen ', 2]
['mulvaney', 3]
['coats', 3]
['gina haspel', 4]
['mnuchin', 5]
['elaine chao', 5]
['mcmahon', 6]
['mike pomeo', 7]
['sonny perdue', 8]
['mattis', 8]
['zinke', 8]
['ben carson', 8]
['wilbur ross', 8]
['devos', 9]
['Robert Lighthizer', 9]
['sessions', 10]
['Robert L. Wilki', 10]
['haley', 10]
['acosta', 11]
['azar', 16]


I scraped each person's opening testimony as a dataset. I then split each person testimony into sentences. I built a model to predict who wrote a sentence given it's features (bigrams of words and characters). 
Applying model to each sentence of resistance article gave a probability that a cabinent member wrote that sentence. 
I set a threshold of .1 for each sentence probability; if a person recieved a probability greater than .1 then they were giving a 1 else 0 for that particular sentence. The above ranking is the sum of those scores. 

