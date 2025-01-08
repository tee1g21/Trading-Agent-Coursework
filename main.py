from agents.testing.test import PerformanceTest
import agents.testing.utils as tutils

from agents.Nathaniel.Dumbass2 import Dumbass2
from agents.Tom.SuperCoolCompany import SuperCoolCompany

from loguru import logger

import json


logger.info("Creating Performance Test")
performance_test = PerformanceTest()
logger.info("Setup Performance Test")
performance_test.setup()

logger.info("Adding Companies")
performance_test.add_company_random_fleet(Dumbass2)
performance_test.add_company_random_fleet(SuperCoolCompany)

logger.info("Running Test")
performance_test.test()
logger.info("End Performance Test")

# performance_test.get_sim_results()


out = None
with open("out/metrics_competition_140025710031168_2025-01-08-02-01-58.json") as f:
    out = json.load(f)

metrics = tutils.MableMetrics(out)

# logger.info(metrics)

logger.info(metrics.companies[0])
logger.info(metrics.companies[1])

# for i in range(len(metrics.company_names)):
#     logger.info(f"{i}: {metrics.company_names[str(i)]}")
