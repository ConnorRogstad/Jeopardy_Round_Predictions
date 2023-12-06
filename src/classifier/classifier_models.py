"""Abstract data type definitions for a basic classifier."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable


__author__ = "Mike Ryu"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Mike Ryu"]
__license__ = "MIT"
__email__ = "mryu@westmont.edu"


class Feature:
    """Feature used classification of an object.

    Attributes:
        _name (str): human-readable name of the feature (e.g., "over 65 years old")
        _value (str): machine-readable value of the feature (e.g., True)
    """

    def __init__(self, name, value=None):
        self._name: str = name
        self._value: Any = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return self._value

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, Feature):
            return False
        else:
            return self._name == other.name and self._value == other.value

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self._name} = {self._value}"

    def __hash__(self) -> int:
        return hash((self._name, self._value))


class FeatureSet:
    """A set of features that represent a single object. Optionally includes the known class of the object.

    Attributes:
        _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
        _clas (str | None): optional attribute set as the pre-defined classification of this object
    """

    def __init__(self, features: set[Feature], known_clas=None):
        self._feat: set[Feature] = features
        self._clas: str | None = known_clas

    @property
    def feat(self):
        return self._feat

    @property
    def clas(self):
        return self._clas

    @classmethod
    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        """Method that builds and returns an instance of FeatureSet given a source object that requires preprocessing.

        For instance, a subclass of `FeatureSet` may be designed to take in a text file object as the `source_object`
        build features based on the tokens that are present in the text file. In this subclass, the logic for
        tokenization and instantiation of `Feature` objects based on the tokens should be written in this method.

        The `return` statement in the actual implementation of this method should simply be a call to the
        constructor where `features` argument is the set of `Feature` instances created within the implementation of
        this method.

        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param kwargs: any additional data needed to preprocess the `source_object` into a feature set
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        pass


class AbstractClassifier(ABC):
    """Abstract definition for an object classifier."""

    @abstractmethod
    def gamma(self, a_feature_set: FeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """
        pass

    @abstractmethod
    def present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        pass

    @classmethod
    @abstractmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        `present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        """
        pass
