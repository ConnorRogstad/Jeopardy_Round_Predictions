"""Query driver for the jeopardy_classifier_models using 200k_questions.json
"""
import importlib.resources
import sys
import json
import random

from jeopardy_classifier_models import *
import re
import string

__author__ = "Connor Rogstad"
__copyright__ = "Copyright 2023, Westmont College, Connor Rogstad"
__credits__ = ["Connor Rogstad"]
__license__ = "MIT"
__email__ = ["crogstad@westmont.edu"]


def main() -> None:

    file_path = '../../data/200k_questions.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # get all questions
    all_questions = data

    # Get the correct classes for the questions
    all_jeopardy_questions = [question for question in all_questions if question["round"] == "Jeopardy!"]
    all_double_jeopardy_questions = [question for question in all_questions if question["round"] == "Double Jeopardy!"]
    all_final_jeopardy_questions = [question for question in all_questions if question["round"] == "Final Jeopardy!"]
    all_tiebreaker_questions = [question for question in all_questions if question["round"] == "Tiebreaker"]

    # build all the feature sets
    all_jeopardy_feature_sets = [JeopardyFeatureSet.build(question, "Jeopardy!")
                                 for question in all_jeopardy_questions]
    all_double_jeopardy_feature_sets = [JeopardyFeatureSet.build(question, "Double Jeopardy!")
                                        for question in all_double_jeopardy_questions]
    all_final_jeopardy_feature_sets = [JeopardyFeatureSet.build(question, "Final Jeopardy!")
                                       for question in all_final_jeopardy_questions]
    all_tiebreaker_feature_sets = [JeopardyFeatureSet.build(question, "Tiebreaker")
                                   for question in all_tiebreaker_questions]

    # combine them
    all_feature_sets = (all_jeopardy_feature_sets + all_double_jeopardy_feature_sets
                        + all_final_jeopardy_feature_sets + all_tiebreaker_feature_sets)
    print("LENGTH OF TOTAL", len(all_feature_sets))

    random.shuffle(all_feature_sets)  # shuffle them so they are not all the same class in a row

    train_jeopardy_feature_sets = all_feature_sets[:173544]  # 80% for training
    test_jeopardy_feature_sets = all_feature_sets[173544:]  # 20% for testing

    our_jeopardy_classifier = JeopardyClassifier.train(train_jeopardy_feature_sets)  # create our classifier

    i = 0
    while i < 10:  # change this to however many we want to see
        print("Actual class: " + test_jeopardy_feature_sets[i].clas + " | Predicted class: "
              + our_jeopardy_classifier.gamma(test_jeopardy_feature_sets[i]))
        i += 1

    print("\nAccuracy = " + str(accuracy(test_jeopardy_feature_sets, 1000, our_jeopardy_classifier)) + "\n")


def accuracy(list_of_sets: list[FeatureSet], amount: int, classifier: AbstractClassifier) -> float:
    i = 0
    accuracy_tally = 0
    while i < amount:  # change this to however many we want to see
        if list_of_sets[i].clas in classifier.gamma(list_of_sets[i]):
            accuracy_tally += 1
        i += 1
    return accuracy_tally / amount


if __name__ == '__main__':
    main()
