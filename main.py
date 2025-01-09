import json
import os

from loguru import logger

import agents.testing.utils as tutils
from agents.Nathaniel.Dumbass2 import Dumbass2
from agents.testing.test import TestingEnvironment
from agents.Tom.SuperCoolCompany import SuperCoolCompany

# logger.info("Creating Performance Test")
# performance_test = PerformanceTest()
# logger.info("Setup Performance Test")
# performance_test.setup()

# logger.info("Adding Companies")
# performance_test.add_company_random_fleet(Dumbass2)
# performance_test.add_company_random_fleet(SuperCoolCompany)

# logger.info("Running Test")
# performance_test.test()
# logger.info("End Performance Test")

# performance_test.get_sim_results()

test_environment = TestingEnvironment()

test_environment.setup_companies([Dumbass2, SuperCoolCompany])
test_environment.setup_random_fleets(1, 1, 1)

test_environment.run_tests(5, 6)

# company_metrics = test_environment.get_test_results("out/test")

# for metric in company_metrics:
#     logger.warning(metric.get_csv_string())

# out = None
# with open("out/metrics_competition_140025710031168_2025-01-08-02-01-58.json") as f:
#     out = json.load(f)

# metrics = tutils.MableMetrics(out)

# logger.info(metrics)

# logger.info("Company Results")
# for company in metrics.companies:
#     logger.info(company)

# for i in range(len(metrics.company_names)):
#     logger.info(f"{i}: {metrics.company_names[str(i)]}")
