import os
import nltk
from nltk.translate.bleu_score import SmoothingFunction

def get_references(rfile):
  if os.path.isfile(rfile):
    references = list()
    with open(rfile) as texts:
      for text in texts:
        text = nltk.word_tokenize(text)
        references.append(text)
    return references
  else:
    print("File not found")
    return 
    
def get_hypothesis(hfile):
  if os.path.isfile(hfile):
    hypothesis = list()
    with open(hfile) as texts:
      for text in texts:
        text = nltk.word_tokenize(text)
        hypothesis.append(text)
    return hypothesis
  else:
    print("File not found")
    return None
def calc_bleu (hfile,rfile,weights):
  res=list()
  references = get_references(rfile)
  hypothesis = get_hypothesis(hfile)
  weight =weights
  for h in hypothesis:
    res.append(nltk.translate.bleu_score.sentence_bleu(references, h, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
  return sum(res)/len(res)
def calc_self_bleu(hfile,weight):
  res = list()
  hypothesis= get_hypothesis(hfile)
  num=len(hypothesis)
  for i in range(num):
    can = hypothesis[i]
    references= hypothesis[:i]+ hypothesis[i+1:]
    res.append(nltk.translate.bleu_score.sentence_bleu(references, can, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
  return sum(res)/len(res)
def get_pos(listToChange):
  res=list()
  for text in listToChange:
    posStr = list();
    for i in nltk.pos_tag(text,'universal'):
      posStr.append(i[1])
    res.append(posStr.copy())
    posStr.clear()
  return res

def calc_pos_bleu(hfile,rfile,weights):
  res=list()
  references = get_pos(get_references(rfile))
  hypothesis = get_pos(get_hypothesis(hfile))
  weight =weights
  for h in hypothesis:
    res.append(nltk.translate.bleu_score.sentence_bleu(references, h, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
  return sum(res)/len(res)
