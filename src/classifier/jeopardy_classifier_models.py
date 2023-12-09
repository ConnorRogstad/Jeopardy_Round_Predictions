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

        category = source_object["category"]  # String of the question's category
        question_text = source_object["question"]  # String of the question's text
        value = source_object["value"]  # String $ amount for the questions value
        answer = source_object["answer"]  # String of the question's answer

        source_set = set()

        # Feature for the Category of Question
        source_set.add(Feature("Category of Question", category))

        # Feature for Length of Question (is it > or < 80)
        if len(question_text) > 80:
            source_set.add(Feature("Amount of characters is >80", True))
        else:
            source_set.add(Feature("Amount of characters is <80", True))

        # Feature for the value of the question
        if value is not None:
            value_int = value[1:]  # get rid of $
            value_int = value_int.replace(",", "")
            value_int = int(value_int)
            source_set.add(Feature("Value of Question", value_int))

        return FeatureSet(source_set, known_clas)


class JeopardyClassifier(AbstractClassifier):
    """Abstract definition for an object classifier."""

    def __init__(self, probability_dict: dict, proportions_list: list):
        self.probability_dict = probability_dict
        self.proportions_list = proportions_list

    def get_probability_dict(self) -> dict:
        return self.probability_dict

    def get_proportions_list(self) -> list:
        return self.proportions_list

    def gamma(self, a_feature_set: FeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """

        gamma_jeopardy = self.proportions_list[0]  # p hat of c
        gamma_double_jeopardy = self.proportions_list[1]  # p hat of c
        gamma_final_jeopardy = self.proportions_list[2]  # p hat of c
        gamma_tiebreaker = self.proportions_list[3]  # p hat of c

        for feature in a_feature_set.feat:
            if self.probability_dict.get(feature, 0) != 0:  # if the feature is in the dictionary
                gamma_jeopardy *= self.probability_dict[feature][0]  # further compute gamma for jeopardy
                gamma_double_jeopardy *= self.probability_dict[feature][1]  # further compute gamma for double_jeopardy
                gamma_final_jeopardy *= self.probability_dict[feature][2]  # further compute gamma for final_jeopardy
                gamma_tiebreaker *= self.probability_dict[feature][3]  # further compute gamma for tiebreaker

        all_gammas = [gamma_jeopardy, gamma_double_jeopardy, gamma_final_jeopardy, gamma_tiebreaker]
        if max(all_gammas) == all_gammas[0]:
            return "Jeopardy!, gamma = " + str(all_gammas[0])
        elif max(all_gammas) == all_gammas[1]:
            return "Double Jeopardy!, gamma = " + str(all_gammas[1])
        elif max(all_gammas) == all_gammas[2]:
            return "Final Jeopardy!, gamma = " + str(all_gammas[2])
        else:
            return "Tiebreaker, gamma = " + str(all_gammas[3])

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

        all_features = {}
        # all_features will be a dict with the feature as its key, and a list of 4 elements as its value
        # the list will represent the predictability of their respective classes:
        # (jeopardy, double_jeopardy, final_jeopardy, tiebreaker)
        # Each num will be the number of that class the feature helped predict /
        # The total number of that class that the feature could have helped predict

        jeopardy_round_tally = 0
        double_jeopardy_round_tally = 0
        final_jeopardy_round_tally = 0
        tiebreaker_tally = 0

        for feature_set in training_set:
            for feature in feature_set.feat:

                if all_features.get(feature, 0) == 0:
                    all_features[feature] = [0, 0, 0, 0]

                if feature_set.clas == "Jeopardy!":
                    all_features[feature][0] += 1
                elif feature_set.clas == "Double Jeopardy!":
                    all_features[feature][1] += 1
                elif feature_set.clas == "Final Jeopardy!":
                    all_features[feature][2] += 1
                else:
                    all_features[feature][3] += 1

            if feature_set.clas == "Jeopardy!":
                jeopardy_round_tally += 1
            elif feature_set.clas == "Double Jeopardy!":
                double_jeopardy_round_tally += 1
            elif feature_set.clas == "Final Jeopardy!":
                final_jeopardy_round_tally += 1
            else:
                tiebreaker_tally += 1

        # divide each # of jeopardy, double_jeopardy, final_jeopardy, or tiebreaker questions with a specific feature by
        # the total number of those class questions respectively
        for feature in all_features.keys():
            all_features[feature][0] /= jeopardy_round_tally
            all_features[feature][1] /= double_jeopardy_round_tally
            all_features[feature][2] /= final_jeopardy_round_tally
            all_features[feature][3] /= tiebreaker_tally

        total_tally = jeopardy_round_tally + double_jeopardy_round_tally + final_jeopardy_round_tally + tiebreaker_tally
        prop_list = [jeopardy_round_tally / total_tally, double_jeopardy_round_tally / total_tally,
                     final_jeopardy_round_tally / total_tally, tiebreaker_tally / total_tally]

        return JeopardyClassifier(all_features, prop_list)
