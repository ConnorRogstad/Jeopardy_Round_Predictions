from src.classifier.classifier_models import *
import json


__author__ = "Connor Rogstad"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Connor Rogstad"]
__license__ = "MIT"
__email__ = ["crogstad@westmont.edu"]


class JeopardyFeature(Feature):
    """JeopardyFeature child class of Feature used specifically for the 200k_questions.json file

    Attributes:
        _name (str): human-readable name of the feature (e.g., "over 65 years old")
        _value (str): machine-readable value of the feature (e.g., True)
    """

    def __init__(self, name, value=None):
        super().__init__(name, value)


class JeopardyFeatureSet(FeatureSet):
    """A set of jeopardy features that represent a single question's features.

    Attributes:
        _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
        _clas (str | None): optional attribute set as the pre-defined classification of this object
    """

    def __init__(self, features: set[Feature], known_clas=None):
        super().__init__(features, known_clas)

    @classmethod
    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        """Method that builds and returns an instance of FeatureSet given a source object that requires preprocessing.

        For instance, a subclass of `FeatureSet` may be designed to take in a text file object as the `source_object`
        build features based on the tokens that are present in the text file. In this subclass, the logic for
        tokenization and instantiation of `Feature` objects based on the tokens should be written in this method.

        The `return` statement in the actual implementation of this method should simply be a call to the
        constructor where `features` argument is the set of `Feature` instances created within the implementation of
        this method.

        :param source_object: a single jeopardy question in json format
        :param known_clas: pre-defined classification of the source object ("Jeopardy!","Double Jeopardy!",
        "Final Jeopardy!" or "Tiebreaker")
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        # TODO: build build

        category = source_object["category"]  # String of the question's category
        question_text = source_object["question"]  # String of the question's text
        value = source_object["value"]  # String $ amount for the questions value
        answer = source_object["answer"]  # String of the question's answer

        source_set = set()

        # Feature for the Category of Question
        source_set.add(Feature("Category of Question", category))

        # Feature for Length of Question (is it > or < 80)
        if len(question_text > 80):
            source_set.add(Feature("Amount of characters is >80", True))
        else:
            source_set.add(Feature("Amount of characters is <80", True))

        # Feature for the value of the question
        value_int = value[1:]  # get rid of $
        value_int = int(value_int)
        source_set.add(Feature("Value of Question", value_int))

        return FeatureSet(source_set, known_clas)


class JeopardyClassifier(AbstractClassifier):
    """Abstract definition for an object classifier."""

    def __init__(self, probability_dict: dict):
        self.probability_dict = probability_dict

    def get_probability_dict(self) -> dict:
        return self.probability_dict

    def gamma(self, a_feature_set: FeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """
        # TODO: create gamma
        gamma_pos = 1/2  # p hat of c
        gamma_neg = 1/2  # p hat of c
        for feature in a_feature_set.feat:
            if self.probability_dict.get(feature, 0) != 0:  # if the feature is in the dictionary
                gamma_pos *= self.probability_dict[feature][0]  # further compute gamma for positive
                gamma_neg *= self.probability_dict[feature][1]  # further compute gamma for negative

        if gamma_pos > gamma_neg:
            return "positive, gamma = " + str(gamma_pos)
        else:
            return "negative, gamma = " + str(gamma_neg)

    def order_features(self, top_n: int = 1) -> str:
        present_dict = dict(self.probability_dict)  # Copy the dictionary

        for feature in present_dict:  # For each feature
            positive_value = present_dict[feature][0]
            negative_value = present_dict[feature][1]
            if positive_value > negative_value:  # If it's more positive than negative
                if negative_value == 0:
                    present_dict[feature] = ["positive : negative", 1]
                else:
                    present_dict[feature] = ["positive : negative", round(positive_value / negative_value, 2)]
            elif positive_value < negative_value:  # If it's more negative than positive
                if positive_value == 0:
                    present_dict[feature] = ["negative : positive", 1]
                else:
                    present_dict[feature] = ["negative : positive", round(negative_value / positive_value, 2)]
            else:  # Just in case...
                present_dict[feature] = ["equal", 1]
            # print(str(present_dict[feature]))

        sortedList = sorted(present_dict.items(), key=lambda item: item[1][1], reverse=True)  # I get lambda now!
        # print("SortedList: " + str(sortedList))
        # sorted() will naturally output a list of this: (FeatureName,["direction", 0.0])
        # the lambda should sort it by our "decimal" value

        returnStr = "Most informative features:"
        index = 0
        while index < top_n:
            space = " " * (3 - len(str(index + 1)))
            featureNameStr = "\n" + str(index + 1) + "." + space + str(sortedList[index][0])
            while len(featureNameStr) < 37:
                featureNameStr += " "
            returnStr += featureNameStr
            returnStr += str(sortedList[index][1][0])
            ratioStr = str(sortedList[index][1][1]) + " : 1"
            while len(ratioStr) < 17:
                ratioStr = " " + ratioStr
            returnStr += ratioStr
            index += 1

        return returnStr

    def present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        # Dictionary format {FeatureName: [PosTally/testSetPos, NegTally/testSetNeg]
        # TODO: create present_features
        print(self.order_features(top_n))

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        `present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
            A SET OF FEATURE SETS (their classifications will be known!)
            An Iterable([FeatureSet]) representation of the Training Set!!!
        :return: an instance of `AbstractClassifier` with its training already completed
        """
        # TODO: create train

        all_features = {}
        # all_features will store each feature that appears in any tweet with the feature name as its key (feature)
        # and it will have the ratio of predicting positive to negative as its value
        # Values will be a list [] with index 0 --> pos and index 1 --> neg

        positive_tally = 0
        negative_tally = 0

        for feature_set in training_set:
            for feature in feature_set.feat:

                if all_features.get(feature, 0) == 0:  # if this feature is already recorded in the dict -> +1 to correct class
                    all_features[feature] = [0, 0]  # adds this feature to the dict

                if feature_set.clas == "positive":
                    all_features[feature][0] += 1  # adds one to class positive
                else:
                    all_features[feature][1] += 1  # adds one to class negative

            # add one to pos or negative total depending on which class the feature_set was
            if feature_set.clas == "positive":
                positive_tally += 1
            else:
                negative_tally += 1

        # divide each # of positive or negative tweets with a specific feature by the total number of positive or
        # negative tweets respectively
        for feature in all_features.keys():
            all_features[feature][0] /= positive_tally
            all_features[feature][1] /= negative_tally

        return TweetClassifier(all_features)
