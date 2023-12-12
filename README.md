# Jeopardy_Round_Predictions

#### Presentation Link: https://docs.google.com/presentation/d/1quLRHS2f-rv3u2Q0OUJZZU8ROOeTNnj-D_GVTZRFDto/edit?pli=1#slide=id.p
#### Author: Connor Rogstad
#### License: MIT License (see file in repo for more details)

## About
This project makes use of Naive Bayes Text Classification techniques to determine which round a Jeopardy question might be from.

## Utilizing my Software
- 

## Citations
- Code:
  The majority of code was written by myself (Connor Rogstad), with contributions coming from Mike Ryu and Davis Peterson.
  - Contributions:
      - jeopardy_classifier_models.py: Connor Rogstad
      - jeopardy_classifier_models_runner.py: Connor Rogstad
      - test_classifier_models.py: Connor Rogstad
      - contributed to the early structure of jeopardy_classifier_models.py: Davis Peterson
      - classifier_models.py: Mike Ryu
      
- Data:
  The dataset used for this assignment was 'Jeopardy Dataset' from Kaggle and provided by Aravind Ram Nathan. The data has a CC0: Public Domain License.
  - This data is stored in a json file with the following attributes:
      - 'category' : the question category, e.g. "HISTORY"
      - 'value' : $ value of the question as string, e.g. "$200"
      - 'question' : text of question
      - 'answer' : text of answer
      - 'round' : one of "Jeopardy!","Double Jeopardy!","Final Jeopardy!" or "Tiebreaker"
      - 'show_number' : string of show number, e.g '4680'
      - 'air_date' : the show air date in format YYYY-MM-DD
