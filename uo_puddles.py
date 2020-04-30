import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from typing import TypeVar, Callable
dframe = TypeVar('pd.core.frame.DataFrame')
narray = TypeVar('numpy.ndarray')


#========================

def hello_ds():
    print("Big hello to you")


#************************************** WEEK 1
def foobar():
  return None

def heat_map(zipped, label_list):
  case_list = []
  for i in range(len(label_list)):
    inner_list = []
    for j in range(len(label_list)):
      inner_list.append(zipped.count((label_list[i], label_list[j])))
    case_list.append(inner_list)


  fig, ax = plt.subplots(figsize=(10, 10))
  ax.imshow(case_list)
  ax.grid(False)
  ax.set_xlabel('Predicted outputs', fontsize=32, color='black')
  ax.set_ylabel('Actual outputs', fontsize=32, color='black')
  ax.xaxis.set(ticks=range(len(label_list)))
  ax.yaxis.set(ticks=range(len(label_list)))
  
  for i in range(len(label_list)):
      for j in range(len(label_list)):
          ax.text(j, i, case_list[i][j], ha='center', va='center', color='white', fontsize=32)
  plt.show()
  return None

def bayes_gothic(evidence:list, evidence_bag:dframe, training_table:dframe, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, list), f'evidence not a list but instead a {type(evidence)}'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'author' in training_table, f'author column is not found in training_table'

  author_list = training_table.author.to_list()
  mapping = ['EAP', 'MWS', 'HPL']
  label_list = [mapping.index(auth) for auth in author_list]
  n_classes = len(set(label_list))
  #assert len(list(evidence_bag.values())[0]) == n_classes, f'Values in evidence_bag do not match number of unique classes ({n_classes}) in labels.'

  word_list = evidence_bag.index.values.tolist()

  evidence = list(set(evidence))  #remove duplicates
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
      if ei not in word_list:
        #did not see word in training set
        the_value =  1/(counts[a_class] + len(evidence_bag))
      else:
        values = evidence_bag.loc[ei].tolist()
        the_value = ((values[a_class]+laplace)/(counts[a_class] + laplace*len(evidence_bag)))
      numerator *= the_value
    #if (numerator * probs[a_class]) == 0: print(evidence)
    results.append(max(numerator * probs[a_class], 2.2250738585072014e-308))

  return tuple(results)

#used week 5 and moved here week 6
def float_mult(number_list: list) -> float:
  assert isinstance(number_list, list), f'number_list should be a list but is instead a {type(number_list)}'
  assert all([isinstance(item, float) for item in number_list]), f'number_list must contain all floats'

  result = 1.
  for number in number_list:  #fancier version of for i in range(n):
    result *= number

  return result

def new_row(table, row_list):
  table.loc[len(table)] = row_list
  return table

def update_gothic_row(word_table, word:str, author:str):
  author_list = ['EAP', 'MWS', 'HPL']
  assert author in author_list, f'{author} not found in {author_list}'
  value_list = [[1,0,0], [0,1,0], [0,0,1]]
  word_list = word_table['word'].tolist()
  real_word = word if type(word) == str else word.text
  k = author_list.index(author)

  if real_word in word_list:
    j = word_list.index(real_word)
    row = word_table.iloc[j].tolist()
    row[1+k] += 1
    word_table.loc[j] = row
  else:
    #not seen yet
    row = [real_word] + value_list[k]
    word_table.loc[len(word_table)] = row
  return word_table

def euclidean_distance(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for euclidean vectors: {len(vect1)} and {len(vect2)}"
  '''
  sum = 0
  for i in range(len(vect1)):
      sum += (vect1[i] - vect2[i])**2
      
  #could put assert here on result   
  return sum**.5  # I claim that this square root is not needed in K-means - see why?
  '''
  a = np.array(vect1)
  b = np.array(vect2)
  return norm(a-b)

def ordered_distances_table(target_vector:list, crowd_table:dframe, answer_column:str, dfunc=euclidean_distance) -> list:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'

  #your code goes here
  crowd_data = crowd_table.drop(answer_column, axis=1) #.drop returns modified deep-copy
  distance_list = [(index, dfunc(target_vector, row.tolist())) for index, row in crowd_data.iterrows()]
  return sorted(distance_list, key=lambda pair: pair[1])

#fix this up at some point - hardwired to skip first 2 items in a row
def knn(target_vector:list, crowd_matrix:list,  labels:list, k:int, sim_type='euclidean') -> int:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_matrix, list), f'crowd_matrix not a list but instead a {type(crowd_matrix)}'

  #assert sim_type in sim_funs, f'sim_type must be one of {list(sim_funs.keys())}.'
    
  if sim_type in ['pearson', 'linear', 'correlation']:
    distance_list = [[index, abs(np.corrcoef(np.array(target_vector), np.array(row))[0][1])] for index,row in enumerate(crowd_matrix)]
    direction = True
  else:
    sim_funs = {'euclidean': [euclidean_distance, False], 'cosine': [cosine_similarity, True]}
    dfunc = sim_funs[sim_type][0]
    distance_list = [[index, dfunc(target_vector, row)] for index,row in enumerate(crowd_matrix)]
    direction = sim_funs[sim_type][1]

  sorted_crowd =  sorted(distance_list, key=lambda pair: pair[1], reverse=direction)  #False is ascending

  #Compute top_k
  top_k = [i for i,d in sorted_crowd[:k]]
  #Compute opinions
  opinions = [labels[index] for index in top_k]
  #Compute winner
  winner = 1 if opinions.count(1) > opinions.count(0) else 0
  #Return winner
  return winner

def ordered_distances_matrix(target_vector:list, crowd_matrix:list,  dfunc=euclidean_distance) -> list:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_matrix, list), f'crowd_matrix not a list but instead a {type(crowd_matrix)}'
  assert all([isinstance(row, list) for row in crowd_matrix]), f'crowd_matrix not a list of lists'
  assert all([len(target_vector)==len(row) for row in crowd_matrix]), f'crowd_matrix has varied row lengths'
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'


  #your code goes here
  distance_list = [(index, dfunc(target_vector, row)) for index, row in enumerate(crowd_matrix)]
  return sorted(distance_list, key=lambda pair: pair[1])

def knn_table(target_vector:list, crowd_table:dframe, answer_column:str, k:int, dfunc:Callable) -> int:
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

def knn_tester(test_table, crowd_table, answer_column, k, dfunc:Callable) -> dict:
  assert isinstance(test_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(test_table)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  
  #your code here
  points = {}
  test_data = test_table.drop(answer_column, axis=1)
  for i in range(len(test_table.index)):
    prediction = knn(test_data.iloc[i].tolist(), crowd_table, answer_column, k, dfunc)
    actual = test_table.iloc[i][answer_column]
    if((prediction, actual) in points):
      points[(prediction, actual)] += 1
    else:
      points[(prediction, actual)] = 1
  return points

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

def fast_cosine(v1:narray, v2:narray) -> float:
  assert isinstance(v1, numpy.ndarray), f"v1 must be a numpy array but instead is {type(v1)}"
  assert len(v1.shape) == 1, f"v1 must be a 1d array but instead is {len(v1.shape)}d"
  assert isinstance(v2, numpy.ndarray), f"v2 must be a numpy array but instead is {type(v2)}"
  assert len(v2.shape) == 1, f"v2 must be a 1d array but instead is {len(v2.shape)}d"
  assert len(v1) == len(v2), f'v1 and v2 must have same length but instead have {len(v1)} and {len(v2)}'

  x = norm(v1)
  if x==0: return 0.0
  y = norm(v2)
  if y==0: return 0.0
  z = x*y
  if z==0: return 0.0  #check for underflow
  return np.dot(v1, v2)/z

def cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"
  '''
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(vect1)):
      x = vect1[i]; y = vect2[i]
      sumxx += x*x
      sumyy += y*y
      sumxy += x*y
      denom = sumxx**.5 * sumyy**.5  #or (sumxx * sumyy)**.5
  #have to invert to order on smallest

  return sumxy/denom if denom > 0 else 0.0
  '''
  return fast_cosine(np.array(vect1), np.array(vect2)).tolist()

def inverse_cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"

  normal_result = cosine_similarity(vect1, vect2)
  return 1.0 - normal_result


#***************************************** WEEK 3

def bayes_laplace(evidence:list, evidence_bag:dframe, training_table:dframe, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, list), f'evidence not a list but instead a {type(evidence)}'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'author' in training_table, f'author column is not found in training_table'

  author_list = training_table.author.to_list()
  mapping = ['EAP', 'MWS', 'HPL']
  label_list = [mapping.index(auth) for auth in author_list]
  n_classes = len(set(label_list))
  #assert len(list(evidence_bag.values())[0]) == n_classes, f'Values in evidence_bag do not match number of unique classes ({n_classes}) in labels.'

  word_list = evidence_bag.index.values.tolist()

  evidence = list(set(evidence))  #remove duplicates
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
      if ei not in word_list:
        #did not see word in training set
        the_value =  1/(counts[a_class] + len(evidence_bag) + laplace)
      else:
        all_values = evidence_bag.loc[ei]
        the_value = ((all_values[a_class]+laplace)/(counts[a_class] + len(evidence_bag) + laplace)) 
      numerator *= the_value

    results.append(max(numerator * probs[a_class], 2.2250738585072014e-308))

  return tuple(results)


def bayes_laplace_tester(testing_table:dframe, evidence_bag:dframe, training_table:dframe, laplace:float=1.0) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'author' in training_table, f'author column is not found in training_table'
  assert 'text' in testing_table, f'text column is not found in testing_table'

  result_list = []
  for i,target_row in testing_table.iterrows():
    raw_text = target_row['text']  #a sentence
    doc = nlp(raw_text.lower())  #create the tokens

    evidence_list = []
    for token in doc:
      if not token.is_alpha or token.is_stop: continue
      evidence_list.append(token.text)

    p_tuple = bayes_laplace(list(set(evidence_list)), evidence_bag, training_table, laplace)
    result_list.append(p_tuple)
  return result_list

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
'''
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

swords = stopwords.words('english')
swords.sort()

import re
def get_clean_words(stopwords:list, raw_sentence:str) -> list:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert all([isinstance(word, str) for word in stopwords]), f'expecting stopwords to be a list of strings'
  assert isinstance(raw_sentence, str), f'raw_sentence must be a list but saw a {type(raw_sentence)}'

  sentence = raw_sentence.lower()
  for word in stopwords:
    sentence = re.sub(r"\b"+word+r"\b", '', sentence)  #replace stopword with empty

  cleaned = re.findall("\w+", sentence)  #now find the words
  return cleaned

def build_word_bag(stopwords:list, training_table:dframe) -> dict:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'

  bow = {}
  starters = [[1,0,0], [0,1,0], [0,0,1]]
  for i,row in training_table.iterrows():
    raw_text = row['text']
    words = set(get_clean_words(stopwords, raw_text))
    label =  row['label']
    for word in words:
        if word in bow:
            bow[word][label] += 1
        else:
            bow[word] = list(starters[label])  #need list to get a copy
  return bow
'''

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
        the_value =  1/(counts[a_class] + len(evidence_bag) + laplace)
      else:
        all_values = evidence_bag[ei]
        the_value = ((all_values[a_class]+laplace)/(counts[a_class] + len(evidence_bag) + laplace)) 
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
    e_set = set(parser(raw_text))
    p_tuple = robust_bayes(e_set, evidence_bag, training_table)
    result_list.append(p_tuple)
  return result_list


