from mable.cargo_bidding import TradingCompany


class MableMetrics:

    def __init__(self, metrics: dict):
        self.company_names = metrics["company_names"]
        self.company_metrics = metrics["company_metrics"]
        self.vessel_metrics = metrics["vessel_metrics"]
        self.global_metrics = metrics["global_metrics"]

        self.company_properties = None

        self.companies: list[CompanyMetric] = []

        for i in range(len(self.company_names)):
            self._extract_company(str(i))

    def get_csv_string(self) -> str:
        out = ""
        if self.company_properties:
            out += '"'
            for company_property in self.company_properties:
                out += f"{company_property[0].__name__}, {company_property[1]},"
            out = out[:-1]
            out += '",'
        for company in self.companies:
            out += company.get_csv_string()
            out += ","
        return out[:-1]

    def set_company_environments(
        self,
        companies_with_fleets: list[tuple[type[TradingCompany], tuple[int, int, int]]],
    ):
        self.company_properties = companies_with_fleets

    def _extract_company(self, key):
        name = self.company_names[key]

        metrics = self.company_metrics[key]

        auctions = []
        for auction in self.global_metrics["auction_outcomes"]:
            auctions += auction[key]

        self.companies.append(CompanyMetric(name, metrics, auctions))

    def __str__(self):
        out = "{{\n"
        out += f"Company Names: {self.company_names},\n"
        out += f"Company Metrics: {self.company_metrics},\n"
        out += f"Vessel Metrics: {self.vessel_metrics},\n"
        out += f"Global Metrics: {self.global_metrics}\n}}"
        return out

    def __repr__(self):
        out = "{{"
        out += f"Company Names: {self.company_names},"
        out += f"Company Metrics: {self.company_metrics},"
        out += f"Vessel Metrics: {self.vessel_metrics},"
        out += f"Global Metrics: {self.global_metrics}}}"
        return out


class CompanyMetric:

    def __init__(self, name: str, company_metrics: str, auctions: list[str]):
        self.name = name
        self.metrics = company_metrics
        self.auctions = auctions

        self.fuel_costs = float(company_metrics["fuel_cost"])

        self.auction_revenue = 0.0
        self.auction_losses = 0.0
        # for auction in auctions:
        for bid in auctions:
            if bid["fulfilled"]:
                self.auction_revenue += float(bid["payment"])
            else:
                self.auction_losses += float(bid["payment"])

        self.profits = self.get_profits()

    def get_csv_string(self):
        return f'"{self.name}, {self.get_profits()}"'
        # return f'"{self.name}, {self.fuel_costs}, {self.auction_revenue}, {self.auction_losses}, {self.get_profits()}"'

    def get_profits(self) -> float:
        return self.auction_revenue - self.auction_losses - self.fuel_costs

    def __str__(self):
        return f"'{self.name}' - {{fuel cost: '{self.fuel_costs}', revenue: '{self.auction_revenue}', losses: '{self.auction_losses}', total: '{self.profits}'}}"

    def __repr__(self):
        return str(self)
