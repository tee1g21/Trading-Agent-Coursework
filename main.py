from loguru import logger

from agents.Nathaniel.Dumbass2 import Dumbass2
from agents.testing.test import TestingEnvironment
from agents.Tom.SuperCoolCompany import SuperCoolCompany

test_environment = TestingEnvironment()

# test_environment.setup_environment(30, 5, 5)

test_environment.setup_companies([Dumbass2, SuperCoolCompany])
test_environment.setup_random_fleets(1, 1, 1)

test_environment.run_tests(5, 6)
