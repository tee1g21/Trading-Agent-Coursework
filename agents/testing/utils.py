
class MableMetrics:

    def __init__(self, metrics: dict):
        self.company_names = metrics["company_names"]
        self.company_metrics = metrics["company_metrics"]
        self.vessel_metrics = metrics["vessel_metrics"]
        self.global_metrics = metrics["global_metrics"]

        self.companies = []

        for i in range(len(self.company_names)):
            self._extract_company(str(i))
    
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
        
        self.profits = self.auction_revenue - self.auction_losses - self.fuel_costs
    
    def __str__(self):
        return f"'{self.name}' - {{fuel cost: '{self.fuel_costs}', revenue: '{self.auction_revenue}', losses: '{self.auction_losses}', total: '{self.profits}'}}"
