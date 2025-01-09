from loguru import logger

from agents.Nathaniel.Dumbass2 import Dumbass2
from agents.testing.test import TestingEnvironment
from agents.Tom.SuperCoolCompany import SuperCoolCompany

# Create our test environment
test_environment = TestingEnvironment()

# test_environment.setup_environment(30, 5, 5)

# Add our companies we want to test
test_environment.setup_companies([Dumbass2, SuperCoolCompany])

# setup our fleet combinations from 0 up to provided numbers
test_environment.setup_random_fleets(1, 1, 1)

# run the tests with a sample of 5 on 6 threads (change to None for sequential runs)
test_environment.run_tests(5, 6)
