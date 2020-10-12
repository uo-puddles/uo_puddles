import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from typing import TypeVar, Callable
dframe = TypeVar('pd.core.frame.DataFrame')
narray = TypeVar('numpy.ndarray')
import math

import json

#===================  fall 20  ===============================

#============ chapter 4

def survival_by_column(table, column, bins=40):
  assert column in table.columns, f'unrecognized column: {column}. Check spelling and case.'
  
  col_pos = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Survived'] == 1]
  col_neg = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Survived'] == 0]
  col_stacked = [col_pos, col_neg]

  import matplotlib.pyplot as plt
  plt.rcParams["figure.figsize"] = (15,8)
  unique = len(table[column].unique())
  if unique <= 20:
    bins = 2*unique - 1
  result = plt.hist(col_stacked, bins, stacked=True, label=['Survived', 'Perished'])
  if unique > 10:
    std = table.std(axis = 0, skipna = True)[column]
    mean = table[column].mean()
    sig3_minus = table[column].min() if (mean-3*std)<=table[column].min() else mean-3*std
    sig3_plus =  mean+3*std
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_minus, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mean, color='k', linestyle='solid', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_plus, color='g', linestyle='dashed', linewidth=1)
  else:
    plt.xticks(table[column].unique().tolist())
    #for label in ax.xaxis.get_xticklabels():
    #  label.set_horizontalalignment('center')
  plt.xlabel(column)
  plt.ylabel('Number of passengers')
  plt.title(f'Survival by {column}')
  plt.legend()
  plt.show()

def survival_by_gender_class(table, a_class):
  assert a_class in table['Class'].to_list(), f'unrecognized class: {a_class}. Check spelling and case.'

  column = 'Gender'
  bins = 3
  col_pos = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Class'] == a_class and table.loc[i, 'Survived'] == 1]
  col_neg = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Class'] == a_class and table.loc[i, 'Survived'] == 0]
  col_stacked = [col_pos, col_neg]

  import matplotlib.pyplot as plt
  plt.rcParams["figure.figsize"] = (15,8)
  result = plt.hist(col_stacked, bins, stacked=True, label=['Survived', 'Perished'])
  if len(table[column].unique()) > 10:
    std = table.std(axis = 0, skipna = True)[column]
    mean = table[column].mean()
    sig3_minus = table[column].min() if (mean-3*std)<=table[column].min() else mean-3*std
    sig3_plus =  mean+3*std
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_minus, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mean, color='k', linestyle='solid', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_plus, color='g', linestyle='dashed', linewidth=1)
  else:
    plt.xticks(table[column].unique().tolist())
    #for label in ax.xaxis.get_xticklabels():
    #  label.set_horizontalalignment('center')
  plt.xlabel(column)
  plt.ylabel('Number of passengers')
  plt.title(f'Survival by {column} cross Class={a_class}')
  plt.legend()
  plt.show()


#survival_by_gender_class(titanic_table, 'C3')


def survival_by_gender_age(table, age_range):
  assert isinstance(age_range, list), f'{age_range} not a list.'
  assert len(age_range)==2, f'{age_range} must be a list of 2 ints.'
  assert isinstance(age_range[0], int), f'{age_range[0]} not an int.'
  assert isinstance(age_range[1], int), f'{age_range[1]} not an int.'

  column = 'Gender'
  bins = 3
  lower = age_range[0]
  upper = age_range[1]
  col_pos = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Age'] >= lower and table.loc[i, 'Age'] <= upper and table.loc[i, 'Survived'] == 1]
  col_neg = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Age'] >= lower and table.loc[i, 'Age'] <= upper and table.loc[i, 'Survived'] == 0]
  col_stacked = [col_pos, col_neg]

  import matplotlib.pyplot as plt
  plt.rcParams["figure.figsize"] = (15,8)
  result = plt.hist(col_stacked, bins, stacked=True, label=['Survived', 'Perished'])
  if len(table[column].unique()) > 10:
    std = table.std(axis = 0, skipna = True)[column]
    mean = table[column].mean()
    sig3_minus = table[column].min() if (mean-3*std)<=table[column].min() else mean-3*std
    sig3_plus =  mean+3*std
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_minus, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mean, color='k', linestyle='solid', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_plus, color='g', linestyle='dashed', linewidth=1)
  else:
    plt.xticks(table[column].unique().tolist())
    #for label in ax.xaxis.get_xticklabels():
    #  label.set_horizontalalignment('center')
  plt.xlabel(column)
  plt.ylabel('Number of passengers')
  plt.title(f'Survival by {column} cross Age={age_range}')
  plt.legend()
  plt.show()


#survival_by_gender_age(titanic_table, [0,10])


def survival_by_class_age(table, age_range):
  assert isinstance(age_range, list), f'{age_range} not a list.'
  assert len(age_range)==2, f'{age_range} must be a list of 2 ints.'
  assert isinstance(age_range[0], int), f'{age_range[0]} not an int.'
  assert isinstance(age_range[1], int), f'{age_range[1]} not an int.'

  column = 'Class'
  bins = 7
  lower = age_range[0]
  upper = age_range[1]
  col_pos = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Age'] >= lower and table.loc[i, 'Age'] <= upper and table.loc[i, 'Survived'] == 1]
  col_neg = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Age'] >= lower and table.loc[i, 'Age'] <= upper and table.loc[i, 'Survived'] == 0]
  col_stacked = [col_pos, col_neg]

  import matplotlib.pyplot as plt
  plt.rcParams["figure.figsize"] = (15,8)
  result = plt.hist(col_stacked, bins, stacked=True, label=['Survived', 'Perished'])
  if len(table[column].unique()) > 10:
    std = table.std(axis = 0, skipna = True)[column]
    mean = table[column].mean()
    sig3_minus = table[column].min() if (mean-3*std)<=table[column].min() else mean-3*std
    sig3_plus =  mean+3*std
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_minus, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mean, color='k', linestyle='solid', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_plus, color='g', linestyle='dashed', linewidth=1)
  else:
    plt.xticks(table[column].unique().tolist())
    #for label in ax.xaxis.get_xticklabels():
    #  label.set_horizontalalignment('center')
  plt.xlabel(column)
  plt.ylabel('Number of passengers')
  plt.title(f'Survival by {column} cross Age={age_range}')
  plt.legend()
  plt.show()


#survival_by_class_age(titanic_table, [30,40])

#============== chapter 5

from sklearn.cluster import KMeans
import numpy as np

def ohe(table, column):
  unique_list = table[column].unique().tolist()
  ohe_cols = [f'ohe_{column}_{val}' for val in unique_list]
  col_matrix = [[0]*len(table) for c in ohe_cols]
  for i in range(len(table)):
    col_val = table.loc[i,column]
    if not isinstance(col_val, str):
      continue
    for j in range(len(ohe_cols)):
      if col_val in ohe_cols[j]:
        col_matrix[j][i] = 1
        break
    else:
      print(f'problem with row {i} and {ohe_cols}')

  ohe_dict = {}
  for k,c in enumerate(ohe_cols):
    ohe_dict[c] = col_matrix[k]
    
  return ohe_dict

#KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, n_jobs='deprecated', algorithm='auto')

def kmeans(k, table, rstate=1234):
  init = 'k-means++'  #be smart about choosing centers
  n_init = 10  #retry with different centers
  max_iter = 300
  tol = 1e-4
  verbose=1
  n_jobs = -1  #use all processors
  kmeans = KMeans(n_clusters=k, random_state=rstate, init=init, n_init=n_init, max_iter=max_iter,
                  tol=tol, verbose=verbose, n_jobs=n_jobs)
  
  numeric_table = table.copy()
  dt = table.dtypes
  ohe_dict = {}  #remember the original columns used for ohe
  for pair in dt.items():
    if pair[1] == np.dtype('O') and len(numeric_table[pair[0]].unique()) <= 10:
      new_cols = ohe(numeric_table, pair[0])
      ohe_dict[pair[0]] = new_cols
      for key in new_cols.keys():
        numeric_table[key] = new_cols[key]
      numeric_table = numeric_table.drop(columns=[pair[0]])
    elif pair[1] == np.dtype('O') and len(numeric_table[pair[0]].unique()) > 10:
      numeric_table = numeric_table.drop(columns=[pair[0]])
    elif pair[1] == np.dtype('float64'):
      numeric_table[pair[0]] = numeric_table[pair[0]]/numeric_table[pair[0]].max()  #normalize
    elif pair[1] == np.dtype('int64') and len(numeric_table[pair[0]].unique()) > 2:
      numeric_table[pair[0]] = numeric_table[pair[0]]/numeric_table[pair[0]].max()  #normalize
  numeric_table = numeric_table.dropna(axis=0)  #remove rows with NaN value
  model = kmeans.fit(numeric_table)
  columns = numeric_table.columns.to_list()
  labels = model.labels_.tolist()

  result_table = pd.DataFrame(columns=columns+['Total'])  #10 x n_cols

  total_cols = len(result_table.columns.to_list())
  for i,center in enumerate(model.cluster_centers_):
    row_dict = dict(zip(columns,center))
    row_dict['Total'] = labels.count(i)
    assert len(row_dict)==total_cols, f'col mismatch {row_dict} and {result_table.columns.to_list()}'
    result_table = result_table.append(row_dict, ignore_index=True)

  #un-normalize and binaryize
  for pair in dt.items():  #use original table column types
    if pair[1] == np.dtype('float64'):
      result_table[pair[0]] = round(result_table[pair[0]] * table[pair[0]].max(),2)
    elif pair[1] == np.dtype('int64') and len(numeric_table[pair[0]].unique()) > 2:
      #print(round(result_table[pair[0]] * table[pair[0]].max(),2))
      s = result_table[pair[0]] * table[pair[0]].max()  #a series
      s = s.round(0) #a series
      s = s.astype(int)
      result_table[pair[0]] = s
    elif pair[1] == np.dtype('int64') and len(numeric_table[pair[0]].unique()) <= 2:
      result_table[pair[0]] = [0 if result_table.loc[i, pair[0]] <= .5 else 1 for i in range(len(result_table))]


  #un-ohe
  drop_cols = []
  for i in range(len(result_table)):
    for original_col in ohe_dict.keys():
      ohe_cols = list(ohe_dict[original_col].keys())  #ohe_Gender_Male, etc.
      ohe_vals = result_table.loc[i, ohe_cols].to_list()
      m = ohe_vals.index(max(ohe_vals))
      name = ohe_cols[m]
      j = name.rfind('_')
      orig_val = name[j+1:]
      result_table.loc[i, original_col] = orig_val
      drop_cols += ohe_cols
  result_table = result_table.drop(columns=drop_cols)
  tot_col = result_table.pop('Total') # remove column Total and store it in df1
  result_table['Total'] = tot_col # add b series as a 'new' column.

  return result_table

#============= chapter 5

def matrix_add(matrix):
  assert isinstance(matrix, list), f'matrix must be a list but instead is a {type(matrix)}'
  assert len(matrix), 'matrix must have at least one row'
  assert isinstance(matrix[0], list), f'matrix must be a list of lists but instead is a list of {type(matrix[0])}'

  new_vec = matrix[0][:]  #get copy of first row
  n = len(new_vec)
  for i in range(1,len(matrix)):
    row = matrix[i]
    for j in range(n):
      new_vec[j] += row[j]
  return new_vec

def row_divide(row, x):
  assert isinstance(row, list), f'row must be a list but instead is a {type(row)}'

  new_vec = [v/x for v in row]
  return new_vec

def rows_to_matrix(table, list_of_rows):
  matrix = table.loc[list_of_rows].values.tolist()
  return matrix

#============= Pima midterm 1

def outcome_by_column(table, column, bins=20):
  col_pos = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Outcome'] == 1]
  col_neg = [table.loc[i, column] for i in range(len(table)) if table.loc[i, 'Outcome'] == 0]
  col_stacked = [col_pos, col_neg]

  import matplotlib.pyplot as plt
  plt.rcParams["figure.figsize"] = (15,8)
  result = plt.hist(col_stacked, bins, stacked=True, label=['Onset within 5 years', 'No onset'])
  if len(table[column].unique()) > 10:
    std = table.std(axis = 0, skipna = True)[column]
    mean = table[column].mean()
    sig3_minus = table[column].min() if (mean-3*std)<=table[column].min() else mean-3*std
    sig3_plus =  mean+3*std
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_minus, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mean, color='k', linestyle='solid', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sig3_plus, color='g', linestyle='dashed', linewidth=1)
  else:
    plt.xticks(table[column].unique().tolist())
    #for label in ax.xaxis.get_xticklabels():
    #  label.set_horizontalalignment('center')
  plt.xlabel(column)
  plt.ylabel('Number of patients')
  plt.title(f'Onset by {column}')
  plt.legend()
  plt.show()

#uncomment when get to it
'''
import spacy, os
os.system('python -m spacy download en_core_web_md')
import en_core_web_md
nlp = en_core_web_md.load()

def nlp_test(s):
      return nlp(s)
'''
    
#======== Below is for week 8 and beyond

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras import Sequential
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import GridSearchCV
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

#libraries to help visualize training results later
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
#%matplotlib inline
rcParams['figure.figsize'] = 10,8

#Used to show progress bar in loop
from IPython.display import HTML, display
import time
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


#from spring 20

def ann_build_model(n:int, layer_list: list, seed=1234, metrics='binary_accuracy'):
  assert isinstance(n, int), f'n is an int, the number of columns/features of each sample. Instead got {type(n)}'
  assert isinstance(layer_list, list) or isinstance(layer_list, tuple), f'layer_list is a list or tuple, the number of nodes per layer. Instead got {type(layer_list)}'

  if len(layer_list) == 1:
    print('Warning: layer_list has only 1 layer, the output layer. So no hidden layers')

  if layer_list[-1] != 1:
    print(f'Warning: layer_list has more than one node in the output layer: {layer_list[-1]}')

  np.random.seed(seed=seed)
  tf.random.set_seed(seed)

  model = Sequential()  #we will always use this in our class. It means left-to-right as we have diagrammed.
  model.add(Dense(units=layer_list[0], activation='sigmoid', input_dim=n))  #first hidden layer needs number of inputs
  for u in layer_list[1:]:
    model.add(Dense(units=u, activation='sigmoid'))

  loss_choice = 'binary_crossentropy'
  optimizer_choice = 'sgd'
  model.compile(loss=loss_choice,
              optimizer=optimizer_choice,
              metrics=[metrics])  #metrics is just to help us to see what is going on. kind of debugging info.
  return model

def ann_train(model, x_train:list, y_train:list, epochs:int,  batch_size=1):
  assert isinstance(x_train, list), f'x_train is a list, the list of samples. Instead got {type(x_train)}'
  assert isinstance(y_train, list), f'y_train is a list, the list of samples. Instead got {type(y_train)}'
  assert len(x_train) == len(y_train), f'x_train must be the same length as y_train'
  assert isinstance(epochs, int), f'epochs is an int, the number of epochs to repeat. Instead got {type(epochs)}'
  assert model.get_input_shape_at(0)[1] == len(x_train[0]), f'model expecting sample size of {model.get_input_shape_at(0)[1]} but saw {len(x_train[0])}'
  
  if epochs == 1:
    print('Warning: epochs is 1, typically too small.')

  xnp = np.array(x_train)
  ynp = np.array(y_train)
  training = model.fit(xnp, ynp, epochs=epochs, batch_size=batch_size, verbose=0)  #3 minutes
  
  plt.plot(training.history['binary_accuracy'])
  plt.plot(training.history['loss'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['binary accuracy', 'loss'], loc='upper left')
  plt.show()
  return training

#for grid search
def create_model(input_dim=300, lyrs=(64,32)):
    model = ann_build_model(input_dim, lyrs, metrics='accuracy')
    return model
  
def grid_search(layers_list, epochs_list, X_train, Y_train, indim=300):
  tup_layers = tuple([tuple(l) for l in layers_list])
  tup_epochs = tuple(epochs_list)
  
  model = KerasClassifier(build_fn=create_model, verbose=0)  #use our create_model
  
  # define the grid search parameters
  batch_size = [1]  #starting with just a few choices
  epochs = tup_epochs
  lyrs = tup_layers

  #use this to override our defaults. keys must match create_model args
  param_grid = dict(batch_size=batch_size, epochs=epochs, input_dim=[indim], lyrs=lyrs)

  # buld the search grid
  grid = GridSearchCV(estimator=model,   #we created model above
                      param_grid=param_grid,
                      cv=3,  #use 3 folds for cross-validation
                      verbose=2)  # include n_jobs=-1 if you are using CPU
  
  grid_result = grid.fit(np.array(X_train), np.array(Y_train))
  
  # summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))
      
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
  assert all([isinstance(item, str) for item in evidence]), f'evidence must be list of strings (not spacy tokens)'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'author' in training_table, f'author column is not found in training_table'

  author_list = training_table.author.unique().tolist()
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

  #CONSIDER CHANGING TO LN OF PRODUCTS. END UP SUMMING LOGS OF EACH ITEM. AVOIDS UNDERFLOW.
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

#uses logs, can handle any number of authors/classes, returns value in slightly different way
def bayes_gothic_gen(evidence:list, evidence_bag:dframe, training_table:dframe, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, list), f'evidence not a list but instead a {type(evidence)}'
  assert all([isinstance(item, str) for item in evidence]), f'evidence must be list of strings (not spacy tokens)'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'author' in training_table, f'author column is not found in training_table'
  
  author_list = sorted(training_table.author.unique().tolist())
  word_list = evidence_bag.index.values.tolist()
  label_list = training_table['author'].tolist()
  evidence = list(set(evidence))  #remove duplicates
  
  counts = []
  probs = []
  for author in author_list:
    ct = label_list.count(author)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes

  results = []
  for i, author in enumerate(author_list):
    prods = [math.log(probs[i])]  #P(author)
    for ei in evidence:
      if ei not in word_list:
        #did not see word in training set
        the_value =  1/(counts[i] + len(evidence_bag))
      else:
        value = evidence_bag.loc[ei, author]
        the_value = ((value+laplace)/(counts[i] + laplace*len(evidence_bag)))
      prods.append(math.log(the_value))
  
    results.append((author, sum(prods)))
  the_min = min(results, key=lambda pair: pair[1])[1]  #shift so smallest is 0
  return [[a,r+abs(the_min)]    for a,r in results]

def naive_bayes(evidence:list, evidence_bag:dframe, training_table:dframe, categorical_column:str, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, list), f'evidence not a list but instead a {type(evidence)}'
  assert all([isinstance(item, str) for item in evidence]), f'evidence must be list of strings (not spacy tokens)'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert categorical_column in training_table, f'categorical_column {categorical_column} is not found in training_table'

  category_list = sorted(training_table[categorical_column].unique().tolist())
  word_list = evidence_bag.index.values.tolist()
  label_list = training_table[categorical_column].tolist()
  evidence = list(set(evidence))  #remove duplicates

  counts = []
  probs = []
  for category in category_list:
    ct = label_list.count(category)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes

  results = []
  for i, category in enumerate(category_list):
    prods = [math.log(probs[i])]  #P(author)
    for ei in evidence:
      if ei not in word_list:
        #did not see word in training set
        the_value =  1/(counts[i] + len(evidence_bag))
      else:
        value = evidence_bag.loc[ei, category]
        the_value = ((value+laplace)/(counts[i] + laplace*len(evidence_bag)))
      prods.append(math.log(the_value))
  
    results.append((category, sum(prods)))
  the_min = min(results, key=lambda pair: pair[1])[1]  #shift so smallest is 0
  return [[a,r+abs(the_min)]    for a,r in results]

#uses logs, can handle any number of authors/classes, returns value in slightly different way
def bayes_by_author(evidence:list, evidence_bag, author_dict, laplace:float=1.0) -> tuple:
  import math
  assert isinstance(evidence, list), f'evidence not a list but instead a {type(evidence)}'
  assert all([isinstance(item, str) for item in evidence]), f'evidence must be list of strings (not spacy tokens)'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(author_dict, dict), f'author_dict not a dict but instead a {type(author_dict)}'
  
  author_list = sorted(list(author_dict.keys()))
  word_list = evidence_bag.index.values.tolist()
  #label_list = training_table['author'].tolist()
  evidence = list(set(evidence))  #remove duplicates

  total_sents = sum(list(author_dict.values()))

  results = []
  for i, author in enumerate(author_list):
    prods = [math.log(author_dict[author]/total_sents)]  #P(author)
    for ei in evidence:
      if ei not in word_list:
        #did not see word in training set
        the_value =  1/(author_dict[author] + len(evidence_bag))
      else:
        value = evidence_bag.loc[ei, author]
        the_value = ((value+laplace)/(author_dict[author] + laplace*len(evidence_bag)))
      prods.append(math.log(the_value))
  
    results.append((author, sum(prods)))
  the_min = min(results, key=lambda pair: pair[1])[1]  #shift so smallest is 0
  return [[a,r+abs(the_min)]    for a,r in results]


def update_gothic_row(word_table, word:str, author:str):
  assert author in word_table.columns.tolist(), f'{author} not found in {word_table.columns.tolist()}'

  word_list = word_table['word'].tolist()
  real_word = word if type(word) == str else word.text

  if real_word in word_list:
    j = word_list.index(real_word)
  else:
    j = len(word_table)
    word_table.loc[j] = [real_word] + [0]*(len(word_table.columns)-1)

  word_table.loc[j, author] += 1

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

def update_tweet_row(word_table, word:str, author:int):
  assert isinstance(author, int), f'Expecting int in author but saw {type(author)}.'
  value_list = [[1,0], [0,1]]
  word_list = word_table['word'].tolist()
  real_word = word if type(word) == str else word.text

  if real_word in word_list:
    j = word_list.index(real_word)
    word_table.loc[j, author] += 1
  else:
    #not seen yet
    row = [real_word] + value_list[author]
    word_table.loc[len(word_table)] = row
  return word_table

def bayes_tweet(evidence:list, evidence_bag, training_table, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, list), f'evidence not a list but instead a {type(evidence)}'
  assert all([isinstance(item, str) for item in evidence]), f'evidence must be list of strings (not spacy tokens)'
  assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'

  label_list = training_table.author.to_list()
  word_list = evidence_bag.index.values.tolist()

  evidence = list(set(evidence))  #remove duplicates
  counts = []
  probs = []
  for i in range(2):
    ct = label_list.count(i)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes
  #CONSIDER CHANGING TO LN OF PRODUCTS. END UP SUMMING LOGS OF EACH ITEM. AVOIDS UNDERFLOW.

  results = []
  for a_class in range(2):
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
#***************************************** WEEK 3



def bayes_gothic_tester(testing_table:dframe, evidence_bag:dframe, training_table:dframe, laplace:float=1.0) -> list:
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
  #CONSIDER CHANGING TO LN OF PRODUCTS. END UP SUMMING LOGS OF EACH ITEM. AVOIDS UNDERFLOW.

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

#embedding stuff
'''
import spacy
spacy.prefer_gpu()  #True if have GPU turned on, False if you just want to run normally

python -m spacy download en_core_web_md  #this fails to load

import en_core_web_md
nlp = en_core_web_md.load()  #Brings in preset vocabulary taken from the web
'''
def word2vec(s:str, nlp) -> list:
    return nlp.vocab[s].vector.tolist()

def subtractv(x:list, y:list) -> list:
  assert isinstance(x, list), f"x must be a list but instead is {type(x)}"
  assert isinstance(y, list), f"y must be a list but instead is {type(y)}"
  assert len(x) == len(y), f"x and y must be the same length"

  #result = [(c1 - c2) for c1, c2 in zip(x, y)]  #one-line compact version - called a list comprehension

  result = []
  for i in range(len(x)):
    c1 = x[i]
    c2 = y[i]
    result.append(c1-c2)

  return result

def addv(x:list, y:list) -> list:
  assert isinstance(x, list), f"x must be a list but instead is {type(x)}"
  assert isinstance(y, list), f"y must be a list but instead is {type(y)}"
  assert len(x) == len(y), f"x and y must be the same length"

  #result = [c1 + c2 for c1, c2 in zip(x, y)]  #one-line compact version

  result = []
  for i in range(len(x)):
    c1 = x[i]
    c2 = y[i]
    result.append(c1+c2)

  return result


def dividev(x:list, c) -> list:
  assert isinstance(x, list), f"x must be a list but instead is {type(x)}"
  assert isinstance(c, int) or isinstance(c, float), f"c must be an int or a float but instead is {type(c)}"

  #result = [v/c for v in x]  #one-line compact version

  result = []
  for i in range(len(x)):
    v = x[i]
    result.append(v/c) #division produces a float

  return result

def meanv(matrix: list) -> list:
    assert isinstance(matrix, list), f"matrix must be a list but instead is {type(x)}"
    assert len(matrix) >= 1, f'matrix must have at least one row'

    #Python transpose: sumv = [sum(col) for col in zip(*matrix)]

    sumv = matrix[0]  #use first row as starting point in "reduction" style
    for row in matrix[1:]:   #make sure start at row index 1 and not 0
      sumv = addv(sumv, row)
    mean = dividev(sumv, len(matrix))
    return mean

def sent2vec(sentence: str, nlp) -> list:
  matrix = []
  doc = nlp(sentence.lower())
  for i in range(len(doc)):
    token = doc[i]
    if token.is_alpha and not token.is_stop:
      vec = token.vector.tolist()
      matrix.append(vec)
  result = [0.0]*300
  if len(matrix) != 0:
    result = meanv(matrix)
  return result

def tokens2vec(tokens: list, stop=True) -> list:
  matrix = []
  for token in tokens:
    if stop and token.is_stop: continue  #skip over stop words
    vec = token.vector.tolist()
    matrix.append(vec)
  result = [0.0]*300  #use this if matrix is empty - all stop words
  if len(matrix) > 0:
    result = meanv(matrix)
  return result

def build_sentence_table(book_dictionary:dict, stop=True):
  assert isinstance(book_dictionary, dict), f'book_dictionary not a dict but a {type(book_dictionary)}'
  all_items = list(book_dictionary.items())
  assert len(all_items) > 0, f'book_dictionary is empty'
  assert all([len(i)==2 for i in all_items]), f'book_dictionary should have a key and single value, i.e., length of an item is 2'
  assert all([isinstance(k, str) for k,v in all_items]), f'all keys must be a string: see key {k}'
  assert all([isinstance(v, str) for k,v in all_items]), f'all values must be a string: see key {k}'

  m = max([len(v)  for k,v in all_items])  #Number of characters in longest book
  old_m = nlp.max_length
  nlp.max_length = m  #for safety
  ordered_sentences = pd.DataFrame(columns = ['text', 'title', 'embedding'])
  for j,item in enumerate(all_items):
    title = item[0]
    raw = item[1]  #the string that contains the entire book
    doc = nlp(raw)  #split into sentences and tokens
    sentences = list(doc.sents)
    print(f'{j+1} of {len(all_items)}, {title}, {len(sentences)} sentences found')
    out = display(progress(0, len(sentences)), display_id=True)  #build new bar for each book
    for i,s in enumerate(sentences):
      tokens = [t for t in s if t.is_alpha or t.is_digit or t.is_punct]
      vec = tokens2vec(tokens, stop)  #averages across all non-stop token vectors
      cleaned_sentence = ' '.join([t.text for t in tokens])
      if cleaned_sentence: ordered_sentences.loc[len(ordered_sentences)] = [cleaned_sentence, title, vec]  #append new row if non-empty
      out.update(progress(i+1, len(sentences)))  #shows progress bar
      time.sleep(0.02)
  nlp.max_length = old_m  #reset to old value
  return ordered_sentences.dropna()  #don't include rows with NaN

def find_most_similar(s:str, sentence_table, stop=True) -> list:
  assert isinstance(s, str), f's should be a string but is insteady a {type(s)}'
  assert isinstance(sentence_table, pd.core.frame.DataFrame), f'sentence_table not a dframe but instead a {type(sentence_table)}'
  columns = sentence_table.columns.to_list()
  assert 'text' in columns, f'cannot find text in columns: {columns}'
  assert 'title' in columns, f'cannot find title in columns: {columns}'
  assert 'embedding' in columns, f'cannot find embedding in columns: {columns}'

  the_text = sentence_table['text'].to_list()  #list of sentences
  the_titles = sentence_table['title'].to_list()  #list of titles
  the_embeddings = sentence_table['embedding'].to_list()  #list of embeddings as list of strings
  real_embeddings = the_embeddings  #do editing at some point

  assert all([isinstance(x, str) for x in the_text]), 'the text column must be all strings'
  assert all([isinstance(x, str) for x in the_titles]), 'the titles column must be all strings'
  assert all([isinstance(x, list) for x in real_embeddings]), 'the embedding column must be all lists'
  assert all([len(x)==300 for x in real_embeddings]), 'an embedding value must be a list of length 300'
  assert all([isinstance(f, float) for em in real_embeddings for f in em]), 'an embedding should be a list of floats'

  target = nlp(s.lower())
  tokens = [t for t in target if (t.is_alpha or t.is_digit or t.is_punct) and (not (stop and t.is_stop))]
  vec = tokens2vec(tokens, stop) if tokens else [0.0]*300
  similarity_list = []
  out = display(progress(0, len(real_embeddings)), display_id=True)
  cut = int(len(real_embeddings)*.2)
  for i,v in enumerate(real_embeddings):
    d = euclidean_distance(vec,v)
    similarity_list.append([i,d, the_text[i], the_titles[i]])
    if i%cut == 0:
      out.update(progress(i+1, len(real_embeddings)))  #shows progress bar
      time.sleep(0.02)
  sim_sorted = sorted(similarity_list, key=lambda p: p[1])
  return sim_sorted

def update_word_table(word_table, word:str, category:str):
  assert category in word_table.columns.tolist(), f'{category} not found in {word_table.columns.tolist()}'
  assert 'word' in word_table.columns.tolist(), f'word not found in {word_table.columns.tolist()}'

  word_list = word_table['word'].tolist()
  real_word = word if type(word) == str else word.text

  if real_word in word_list:
    j = word_list.index(real_word)
  else:
    j = len(word_table)
    word_table.loc[j] = [real_word] + [0]*(len(word_table.columns)-1)

  word_table.loc[j, category] += 1

  return word_table

def build_word_table(books:dict):
  assert isinstance(books, dict), f'books not a dictionary but instead a {type(books)}'

  all_titles = list(books.keys())
  n = len(all_titles)
  word_table = pd.DataFrame(columns=['word'] + all_titles)
  m = max([len(v)  for v in books.values()])  #Number of characters in longest book
  nlp.max_length = m

  for i,title in enumerate(all_titles):
    print(f'({i+1} of {n}) Processing {title} ({len(books[title])} characters)')
    doc = nlp(books[title].lower()) #parse the entire book into tokens
    out = display(progress(0, len(doc)), display_id=True)
    cut = int(len(doc)*.1)
    for j,token in enumerate(doc):
      if  token.is_alpha and not token.is_stop:
        word_table = update_word_table(word_table, token.text, title)
      if j%cut==0:
        out.update(progress(j+1, len(doc)))  #shows progress bar
        time.sleep(0.02)

  word_table = word_table.infer_objects()
  #word_table = word_table.astype(int)  #all columns
  word_table = word_table.astype({'word':str})  #now just word column

  sorted_word_table = word_table.sort_values(by=['word'])
  sorted_word_table = sorted_word_table.reset_index(drop=True)
  sorted_word_table = sorted_word_table.set_index('word')  #set the word column to be the table index

  return sorted_word_table

def most_similar_word(word_table, target_word:str) -> list:
  assert isinstance(word_table, pd.core.frame.DataFrame), f'word_table not a dframe but instead a {type(word_table)}'

  target_vec = list(nlp.vocab.get_vector(target_word))
  distance_list = []
  word_list = word_table.index.to_list()
  out = display(progress(0, len(word_list)), display_id=True)
  cut = int(len(word_list)*.1)
  for i,word in enumerate(word_list):
    vec = list(nlp.vocab.get_vector(word))
    d = euclidean_distance(target_vec, vec)
    distance_list.append([word, d])
    if i%cut==0:
      out.update(progress(i+1, len(word_list)))  #shows progress bar
      time.sleep(0.02)
  ordered = sorted(distance_list, key=lambda p: p[1])
  return ordered
