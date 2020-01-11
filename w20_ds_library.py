import pandas as pd
from typing import TypeVar, Callable
dframe = TypeVar('pd.core.frame.DataFrame')

def hello_ds():
    print("Big hello to you")
    
#************************************** WEEK 1

def euclidean_distance(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for euclidean vectors: {len(vect1)} and {len(vect2)}"

  sum = 0
  for i in range(len(vect1)):
      sum += (vect1[i] - vect2[i])**2
      
  #could put assert here on result   
  return sum**.5  # I claim that this square root is not needed in K-means - see why?

def ordered_distances(target_vector:list, crowd_table:dframe, answer_column:str, dfunc:Callable) -> list:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'

  distance_list = []
  for i,crow in crowd_table.iterrows():
    c_vector = crow.drop([answer_column], axis=0).tolist()
    d = dfunc(target_vector, c_vector)
    distance_list.append((i,d))
  return sorted(distance_list, key=lambda pair: pair[1], reverse=False)

def knn(target_vector:list, crowd_table:dframe, answer_column:str, k:int, dfunc:Callable) -> int:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'

  #Comute sorted_crowd
  sorted_crowd = ordered_distances(target_vector, crowd_table, answer_column, dfunc)
  #Compute top_k
  top_k = [i for i,d in sorted_crowd[:k]]
  #Compute opinions
  opinions = [crowd_table.iloc[i][answer_column] for i in top_k]
  #Compute winner
  winner = 1 if opinions.count(1) > opinions.count(0) else 0
  #Return winner
  return winner

def knn_tester(test_table:dframe, crowd_table:dframe, answer_column:str, k:int, dfunc:Callable) -> float:
  assert isinstance(test_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(test_table)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
    
  confusion_dictionary = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
  correct = 0
  for i,target_row in test_table.iterrows():
    target_vector = target_row.drop([answer_column], axis=0).tolist()
    crowd_answer = knn(target_vector, crowd_table, answer_column, k, euclidean_distance)
    real_answer = target_row[answer_column]
    confusion_dictionary[(crowd_answer, real_answer)] += 1
  return confusion_dictionary

def cm_accuracy(confusion_dictionary: dict) -> float:
  assert isinstance(confusion_dictionary, dict), f'confusion_dictionary not a dictionary but instead a {type(confusion_dictionary)}'
  
  tp = confusion_dictionary[(1,1)]
  fp = confusion_dictionary[(1,0)]
  fn = confusion_dictionary[(0,1)]
  tn = confusion_dictionary[(0,0)]
  
  return (tp+tn)/(tp+fp+fn+tn)

def cm_f1(confusion_dictionary: dict) -> float:
  assert isinstance(confusion_dictionary, dict), f'confusion_dictionary not a dictionary but instead a {type(confusion_dictionary)}'
  
  tp = confusion_dictionary[(1,1)]
  fp = confusion_dictionary[(1,0)]
  fn = confusion_dictionary[(0,1)]
  tn = confusion_dictionary[(0,0)]
  
  recall = tp/(tp+fn) if (tp+fn) != 0 else 0  #Heuristic
  precision = tp/(tp+fp) if (tp+fp) != 0 else 0  #Heuristic
  
  recall_div = 1/recall if recall != 0 else 0  #Heuristic
  precision_div = 1/precision if precision != 0 else 0  #Heuristic
  
  f1 = 2/(recall_div+precision_div) if (recall_div+precision_div) != 0 else 0  #heuristic
  
  return f1

#************************************** WEEK 2

def cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"
  
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(vect1)):
      x = vect1[i]; y = vect2[i]
      sumxx += x*x
      sumyy += y*y
      sumxy += x*y
      denom = sumxx**.5 * sumyy**.5  #or (sumxx * sumyy)**.5
  #have to invert to order on smallest

  return sumxy/denom if denom > 0 else 0.0

def inverse_cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"

  normal_result = cosine_similarity(vect1, vect2)
  return 1.0 - normal_result


#***************************************** WEEK 3

def bayes(evidence:set, evidence_bag:dict, training_table:dframe) -> tuple:
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"

  label_list = training_table.label.to_list()
  n_classes = len(set(label_list))
  assert len(list(evidence_bag.values())[0]) == n_classes, f'Values in evidence_bag do not match number of unique classes ({n_classes}) in labels.'

  counts = []
  probs = []
  for i in range(n_classes):
    ct = label_list.count(i)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes

  results = []
  for a_class in range(n_classes):
    numerator = 1
    for ei in evidence:
      all_values = evidence_bag[ei]
      the_value = (all_values[a_class]/counts[a_class])
      numerator *= the_value
    results.append(numerator * probs[a_class])

  return tuple(results)

def char_set_builder(text:str) -> list:
  the28 = set(text).intersection(set('abcdefghijklmnopqrstuvwxyz!#'))
  return list(the28)

def bayes_tester(testing_table:dframe, evidence_bag:dict, training_table:dframe, parser:Callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'


  result_list = []
  for i,target_row in testing_table.iterrows():
    raw_text = target_row['text']
    e_set = set(parser(raw_text))
    p_tuple = bayes(e_set, evidence_bag, training_table)
    result_list.append(p_tuple)
  return result_list

#***************************************** WEEK 4

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

swords = stopwords.words('english')
swords.sort()

import re
def get_clean_words(stopwords:list, raw_sentence:str) -> set:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert all([isinstance(word, str) for word in stopwords]), f'expecting stopwords to be a list of strings'
  assert isinstance(raw_sentence, str), f'raw_sentence must be a list but saw a {type(raw_sentence)}'

  sentence = raw_sentence.lower()
  for word in stopwords:
    sentence = re.sub(r"\b"+word+r"\b", '', sentence)  #replace stopword with empty

  cleaned = re.findall("\w+", sentence)  #now find the words
  return set(cleaned)

def build_word_bag(stopwords:list, training_table:dframe) -> dict:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'

  bow = {}
  starters = [[1,0,0], [0,1,0], [0,0,1]]
  for i,row in training_table.iterrows():
    raw_text = row['text']
    words = get_clean_words(stopwords, raw_text)
    label =  row['label']
    for word in words:
        if word in bow:
            bow[word][label] += 1
        else:
            bow[word] = list(starters[label])  #need list to get a copy
  return bow

def robust_bayes(evidence:set, evidence_bag:dict, training_table:dframe, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"

  label_list = training_table.label.to_list()
  n_classes = len(set(label_list))
  assert len(list(evidence_bag.values())[0]) == n_classes, f'Values in evidence_bag do not match number of unique classes ({n_classes}) in labels.'

  counts = []
  probs = []
  for i in range(n_classes):
    ct = label_list.count(i)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes

  results = []
  for a_class in range(n_classes):
    numerator = 1
    for ei in evidence:
      if ei not in evidence_bag:
        the_value =  1/(counts[a_class] + len(training_table) + laplace)
      else:
        all_values = evidence_bag[ei]
        the_value = ((all_values[a_class]+laplace)/(counts[a_class] + len(training_table) + laplace)) 
      numerator *= the_value
    results.append(max(numerator * probs[a_class], 2.2250738585072014e-308))

  return tuple(results)

def robust_bayes_tester(testing_table:dframe, evidence_bag:dict, training_table:dframe, parser:Callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'

  result_list = []
  for i,target_row in testing_table.iterrows():
    raw_text = target_row['text']
    e_set = parser(raw_text)
    p_tuple = robust_bayes(e_set, evidence_bag, training_table)
    result_list.append(p_tuple)
  return result_list

def curry_x(old_func,x):
  #this is the closure. It contains the value of x.

  def inner(y): return old_func(x,y)  #define a new function that uses x from closure and passes in y.

  return inner  #will return a function of 1 argument. But that function will carry the closure with it.


word_parser = curry_x(get_clean_words, swords)
