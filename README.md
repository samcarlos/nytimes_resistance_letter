# nytimes_resistance_letter
Which senior cabinet wrote nytimes 'resistance' article? 

Below ranking of each cabinent member. Azar, Acosta, or Haley are most likely to have written article based on data analysis. 


['sonny perdue', 0.042799391346238246]
['rick perry', 0.042827496150054725]
['kirstjen nielsen ', 0.04348531255720842]
['Kelly_John', 0.04388015088869095]
['gina haspel', 0.04408884887851884]
['mulvaney', 0.044449131042627626]
['mattis', 0.044520551404359426]
['zinke', 0.04460915299172605]
['mnuchin', 0.04475626096821514]
['mike pomeo', 0.04489294864505752]
['coats', 0.04504337727245659]
['wilbur ross', 0.0451681578002181]
['acosta', 0.045206788220377866]
['mcmahon', 0.04568208831113954]
['Robert Lighthizer', 0.04569583598035352]
['elaine chao', 0.04579825015621833]
['haley', 0.045990404889975765]
['devos', 0.04708158774824911]
['azar', 0.047256249688986826]
['ben carson', 0.04796853780460778]
['Robert L. Wilki', 0.0486808466773789]
['sessions', 0.05011863057734076]

I scraped each person's opening testimony as a dataset. I then split each person testimony into sentences. I built a model to predict who wrote a sentence given it's features (bigrams of words and characters). 
Applying model to each sentence of resistance article gave a probability that a cabinent member wrote that sentence. 
The above ranking is the mean of those scores. 

