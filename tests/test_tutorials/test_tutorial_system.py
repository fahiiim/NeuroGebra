"""Tests for TutorialSystem."""

import pytest
from neurogebra.tutorials.tutorial_system import (
    TutorialSystem,
    BasicsTutorial,
    FirstModelTutorial,
    ActivationsTutorial,
    TrainingTutorial,
)


class TestTutorialSystemInit:
    """Test TutorialSystem initialization."""

    def test_creates_successfully(self):
        """Test TutorialSystem creates successfully."""
        ts = TutorialSystem()
        assert ts is not None

    def test_has_tutorials(self):
        """Test system has all expected tutorials."""
        ts = TutorialSystem()
        assert "basics" in ts.tutorials
        assert "first_model" in ts.tutorials
        assert "activations" in ts.tutorials
        assert "training" in ts.tutorials

    def test_tutorial_count(self):
        """Test the number of tutorials."""
        ts = TutorialSystem()
        assert len(ts.tutorials) == 4


class TestTutorialSystemMenu:
    """Test showing tutorial menu."""

    def test_show_menu(self, capsys):
        """Test show_menu prints available tutorials."""
        ts = TutorialSystem()
        ts.show_menu()
        captured = capsys.readouterr()
        assert "basics" in captured.out
        assert "first_model" in captured.out
        assert "activations" in captured.out
        assert "training" in captured.out

    def test_show_menu_has_header(self, capsys):
        """Test menu has a header."""
        ts = TutorialSystem()
        ts.show_menu()
        captured = capsys.readouterr()
        assert "Neurogebra" in captured.out or "Tutorial" in captured.out


class TestTutorialSystemStart:
    """Test starting tutorials."""

    def test_start_basics(self, capsys):
        """Test starting basics tutorial."""
        ts = TutorialSystem()
        ts.start("basics")
        captured = capsys.readouterr()
        assert "Neural Network" in captured.out

    def test_start_first_model(self, capsys):
        """Test starting first_model tutorial."""
        ts = TutorialSystem()
        ts.start("first_model")
        captured = capsys.readouterr()
        assert "Model" in captured.out or "model" in captured.out

    def test_start_activations(self, capsys):
        """Test starting activations tutorial."""
        ts = TutorialSystem()
        ts.start("activations")
        captured = capsys.readouterr()
        assert "Activation" in captured.out

    def test_start_training(self, capsys):
        """Test starting training tutorial."""
        ts = TutorialSystem()
        ts.start("training")
        captured = capsys.readouterr()
        assert "Training" in captured.out

    def test_start_invalid(self, capsys):
        """Test starting invalid tutorial shows error."""
        ts = TutorialSystem()
        ts.start("nonexistent")
        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestBasicsTutorial:
    """Test BasicsTutorial content."""

    def test_run(self, capsys):
        """Test basics tutorial runs to completion."""
        tutorial = BasicsTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "STEP 1" in captured.out
        assert "STEP 2" in captured.out
        assert "STEP 3" in captured.out
        assert "STEP 4" in captured.out
        assert "Tutorial complete" in captured.out or "complete" in captured.out

    def test_covers_neurons(self, capsys):
        """Test tutorial covers neurons."""
        tutorial = BasicsTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "neuron" in captured.out.lower() or "Neuron" in captured.out


class TestFirstModelTutorial:
    """Test FirstModelTutorial content."""

    def test_run(self, capsys):
        """Test first model tutorial runs to completion."""
        tutorial = FirstModelTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "STEP 1" in captured.out
        assert "Congratulations" in captured.out or "complete" in captured.out.lower()

    def test_covers_compile(self, capsys):
        """Test tutorial covers compilation."""
        tutorial = FirstModelTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "compile" in captured.out.lower()


class TestActivationsTutorial:
    """Test ActivationsTutorial content."""

    def test_run(self, capsys):
        """Test activations tutorial runs to completion."""
        tutorial = ActivationsTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "ReLU" in captured.out
        assert "Sigmoid" in captured.out

    def test_covers_choosing(self, capsys):
        """Test tutorial covers how to choose."""
        tutorial = ActivationsTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "Choose" in captured.out or "choose" in captured.out


class TestTrainingTutorial:
    """Test TrainingTutorial content."""

    def test_run(self, capsys):
        """Test training tutorial runs to completion."""
        tutorial = TrainingTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "Loss" in captured.out or "loss" in captured.out
        assert "Optimizer" in captured.out or "optimizer" in captured.out

    def test_covers_evaluation(self, capsys):
        """Test tutorial covers evaluation."""
        tutorial = TrainingTutorial()
        tutorial.run()
        captured = capsys.readouterr()
        assert "Evaluat" in captured.out
