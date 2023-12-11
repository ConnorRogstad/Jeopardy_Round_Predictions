import unittest
from src.classifier.jeopardy_classifier_models import *

__author__ = "Connor Rogstad"
__copyright__ = "Copyright 2023, Westmont College, Connor Rogstad"
__credits__ = ["Connor Rogstad"]
__license__ = "MIT"
__email__ = ["crogstad@westmont.edu"]


class JeopardyFeatureSetTest(unittest.TestCase):

    def setUp(self):
        question1 = {"category": "HISTORY", "air_date": "2004-12-31", "question": "'For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory'",
                     "value": "$200", "answer": "Copernicus", "round": "Jeopardy!", "show_number": "4680"}
        question2 = {"category": "PRESIDENTIAL STATES OF BIRTH", "air_date": "2004-12-31", "question": "'California'",
                     "value": "$400", "answer": "Nixon", "round": "Double Jeopardy!", "show_number": "4680"}

        self.actual_jeopardy_feature_set1 = JeopardyFeatureSet.build(question1, "Jeopardy!")
        self.actual_jeopardy_feature_set2 = JeopardyFeatureSet.build(question2, "Double Jeopardy!")

        self.feature1 = JeopardyFeature("Category of Question", "HISTORY")
        self.feature2 = JeopardyFeature("Amount of characters is >80", True)

        self.feature3 = JeopardyFeature("Value of Question", 400)
        self.feature4 = JeopardyFeature("Amount of characters is <80", True)

    def test_build_category(self):
        self.assertIn(self.feature1, self.actual_jeopardy_feature_set1.feat)

    def test_build_value(self):
        self.assertIn(self.feature3, self.actual_jeopardy_feature_set2.feat)

    def test_build_length(self):
        self.assertIn(self.feature2, self.actual_jeopardy_feature_set1.feat)
        self.assertIn(self.feature4, self.actual_jeopardy_feature_set2.feat)


class JeopardyClassifierTest(unittest.TestCase):

    def setUp(self):
        question1 = {"category": "HISTORY", "air_date": "2004-12-31", "question": "'For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory'",
                     "value": "$200", "answer": "Copernicus", "round": "Jeopardy!", "show_number": "4680"}
        question2 = {"category": "PRESIDENTIAL STATES OF BIRTH", "air_date": "2004-12-31", "question": "'California'",
                     "value": "$400", "answer": "Nixon", "round": "Double Jeopardy!", "show_number": "4680"}
        question3 = {"category": "HOMOPHONIC PAIRS", "air_date": "1996-12-06", "question": "'A squash that's been pierced by a bull's horn'",
                     "value": "$400", "answer": "gored gourd", "round": "Jeopardy!", "show_number": "2825"}
        question4 = {"category": "BRITISH NOVELS", "air_date": "1996-12-06", "question": "'This 1895 novel is subtitled \"An Invention\"'",
                     "value": None, "answer": "The Time Machine", "round": "Final Jeopardy!", "show_number": "2825"}
        question5 = {"category": "CHILD'S PLAY", "air_date": "2007-11-13", "question": "'A Longfellow poem & a Lillian Hellman play about a girls' boarding school share this timely title'",
                     "value": None, "answer": "The Children\\'s Hour", "round": "Tiebreaker", "show_number": "5332"}

        self.jeopardy_class_questions = [question1, question2, question3, question4, question5]
        # self.double_jeopardy_class_questions = [question2, question3, question4]  # more double jeopardy class questions

        # all features for jeopardy class questions
        self.feature1 = JeopardyFeature("Category of Question", "HISTORY")
        self.feature2 = JeopardyFeature("Amount of characters is >80", True)
        self.feature3 = JeopardyFeature("Value of Question", 200)
        self.feature4 = JeopardyFeature("Category of Question", "PRESIDENTIAL STATES OF BIRTH")
        self.feature5 = JeopardyFeature("Value of Question", 400)
        self.feature6 = JeopardyFeature("Amount of characters is <80", True)
        self.feature7 = JeopardyFeature("Category of Question", "HOMOPHONIC PAIRS")
        self.feature8 = JeopardyFeature("Value of Question", 400)
        self.feature9 = JeopardyFeature("Amount of characters is <80", True)
        self.feature10 = JeopardyFeature("Category of Question", "BRITISH NOVELS")
        self.feature11 = JeopardyFeature("Amount of characters is <80", True)
        self.feature12 = JeopardyFeature("Category of Question", "CHILD'S PLAY")
        self.feature13 = JeopardyFeature("Amount of characters is >80", True)
        self.set1 = {self.feature1, self.feature2, self.feature3}
        self.set2 = {self.feature4, self.feature5, self.feature6}
        self.set3 = {self.feature7, self.feature8, self.feature9}
        self.set4 = {self.feature10, self.feature11}
        self.set5 = {self.feature12, self.feature13}
        self.feature_set1 = JeopardyFeatureSet(self.set1, "Jeopardy!")
        self.feature_set2 = JeopardyFeatureSet(self.set2, "Double Jeopardy!")
        self.feature_set3 = JeopardyFeatureSet(self.set3, "Jeopardy!")
        self.feature_set4 = JeopardyFeatureSet(self.set4, "Final Jeopardy!")
        self.feature_set5 = JeopardyFeatureSet(self.set5, "Tiebreaker")
        self.jeopardy_class_feature_sets = [self.feature_set1, self.feature_set2, self.feature_set3,
                                            self.feature_set4, self.feature_set5]

        self.trained_classifier = JeopardyClassifier.train(self.jeopardy_class_feature_sets)

        self.constructed_prob_dict = {
            self.feature2: [0.5, 0.0, 0.0, 1.0],
            self.feature1: [0.5, 0.0, 0.0, 0.0],
            self.feature3: [0.5, 0.0, 0.0, 0.0],
            self.feature4: [0.0, 1.0, 0.0, 0.0],
            self.feature6: [0.5, 1.0, 1.0, 0.0],
            self.feature5: [0.5, 1.0, 0.0, 0.0],
            self.feature7: [0.5, 0.0, 0.0, 0.0],
            self.feature10: [0.0, 0.0, 1.0, 0.0],
            self.feature12: [0.0, 0.0, 0.0, 1.0],
        }
        self.constructed_classifier = JeopardyClassifier(self.constructed_prob_dict, [])

    def test_train(self):
        my_dict = self.trained_classifier.get_probability_dict()

        self.assertEqual(my_dict, self.constructed_classifier.get_probability_dict())

    def test_gamma_jeopardy(self):
        output1 = self.trained_classifier.gamma(self.feature_set1)

        self.assertEqual(output1, "Jeopardy!, gamma = " + str(1/20))

    def test_gamma_double_jeopardy(self):
        output2 = self.trained_classifier.gamma(self.feature_set2)

        self.assertEqual(output2, "Double Jeopardy!, gamma = " + str(1/5))

    def test_gamma_final_jeopardy(self):
        output3 = self.trained_classifier.gamma(self.feature_set4)

        self.assertEqual(output3, "Final Jeopardy!, gamma = " + str(1/5))

    def test_gamma_tiebreaker(self):
        output4 = self.trained_classifier.gamma(self.feature_set5)

        self.assertEqual(output4, "Tiebreaker, gamma = " + str(1/5))


if __name__ == '__main__':
    unittest.main()
