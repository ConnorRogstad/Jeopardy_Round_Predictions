# Jeopardy_Round_Predictions

## About
This project makes use of Naive Bayes Text Classification techniques to determine which round a Jeopardy question might be from.

## Citations
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
