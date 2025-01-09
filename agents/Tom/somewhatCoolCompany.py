from mable.cargo_bidding import TradingCompany
from mable.simulation_space.universe import OnJourney
from mable.transport_operation import ScheduleProposal

import attrs
from marshmallow import fields


# -------------------[PRINT FORMATTERS]--------------------
def fmt_location(location) -> str:
    if isinstance(location, OnJourney):
        return f"DST: ({location.destination.x}, {location.destination.y})"

    return f"{location.name}: ({location.x}, {location.y})"


def fmt_trade(trade) -> str:
    return f"[{fmt_location(trade.origin_port)}] => [{fmt_location(trade.destination_port)}]"


def fmt_vessel(vessel) -> str:
    return f"[{vessel.name} @ {fmt_location(vessel.location)}]"


class SuperCoolCompany(TradingCompany):
    
    def __init__(self, fleet, name, profit_factor=1.65):
        """
        :param fleet: The companies fleet.
        :param name: The companies name.
        :param profit_factor: The companies profit factor, i.e. factor applied to cost to determine bids.
        """
        super().__init__(fleet, name)
        self._profit_factor = profit_factor
        self.company_profit_factors = {}
        
    @attrs.define
    class Data(TradingCompany.Data):
        profit_factor: float = 1.65

        class Schema(TradingCompany.Data.Schema):
            profit_factor = fields.Float(default=1.65)
        
    def pre_inform(self, trades, time):
        return super().pre_inform(trades, time)
    
    def inform(
            self,
            trades,
            *args, **kwargs
    ):
        """
        auction started, propose bids
        """

        bids = super().inform(trades, *args, **kwargs)

        if len(bids) == 0:
            self.log("no bids")
        else:
            self.log("----- bids -----")
            for bid in bids:
                self.log(f"{fmt_trade(bid.trade)}: {bid.amount}")
            self.log("----------------")

        return bids
    
    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        
        # initialise default profit factors for missing companies
        for company in self.headquarters.get_companies():
           if company.name not in self.company_profit_factors:
               self.company_profit_factors[company.name] = [1.2]
               
        # Process each company's won contracts
        for company_name, won_contracts in auction_ledger.items():
            company_fleet = [c for c in self.headquarters.get_companies() if c.name == company_name].pop().fleet

            for contract in won_contracts:
                trade = contract.trade
                actual_payment = contract.payment

                # Step 1: Predict payment using the current profit factor
                predicted_payment = self.predict_payment(company_name, trade)

                # Step 2: Calculate and store the profit factor
                best_cost = float('inf')
                for vessel in company_fleet:
                    predicted_cost = self.predict_cost(vessel, trade)
                    if predicted_cost < best_cost:
                        best_cost = predicted_cost

                if best_cost > 0:
                    profit_factor = actual_payment / best_cost
                else:
                    profit_factor = float('inf')

                self.profit_factors[company_name].append(profit_factor)

                # Step 3: Adjust predictions dynamically
                if predicted_payment is not None:
                    self.adjust_predictions(company_name, actual_payment, predicted_payment)

                # Debugging
                print(f"Company: {company_name}")
                print(f"  Trade: {trade.origin_port.name} -> {trade.destination_port.name}")
                print(f"  Actual Payment: {actual_payment}, Predicted Payment: {predicted_payment}")
                print(f"  Adjustment Factor: {actual_payment / predicted_payment if predicted_payment > 0 else 'N/A'}")

            super().receive(contracts, auction_ledger, *args, **kwargs)
    
    def adjust_predictions(self, company_name, actual_payment, predicted_payment):
        if company_name not in self.profit_factors:
            self.profit_factors[company_name] = [1.2]  # Default value if not initialized

        # Calculate adjustment factor
        adjustment_factor = actual_payment / predicted_payment if predicted_payment > 0 else float('inf')

        # Update profit factor using a weighted moving average
        previous_factors = self.profit_factors[company_name]
        updated_factor = (sum(previous_factors) + adjustment_factor) / (len(previous_factors) + 1)
        
        self.profit_factors[company_name] = [updated_factor]  # Replace the list with the single updated value

        # Debug output
        print(f"Updated profit factor for {company_name}: {updated_factor}")

    
    def predict_payment(self, company_name, trade):
        if company_name not in self.profit_factors or not self.profit_factors[company_name]:
            return None  # No data available for prediction

        # Use the average profit factor
        average_profit_factor = sum(self.profit_factors[company_name]) / len(self.profit_factors[company_name])

        # Predict cost for the trade
        best_cost = float('inf')
        for vessel in self.fleet:
            predicted_cost = self.predict_cost(vessel, trade)
            if predicted_cost < best_cost:
                best_cost = predicted_cost

        # Predicted payment is based on the average profit factor
        predicted_payment = average_profit_factor * best_cost
        return predicted_payment
    
    def predict_bid(self, company_name, trade, multiplier=0.95):
        predicted_payment = self.predict_payment(company_name, trade)
        if predicted_payment is None:
            return None  # No data available for prediction

        # Assume bid is slightly below predicted payment
        predicted_bid = predicted_payment * multiplier  # Adjust multiplier based on observed behavior
        return predicted_bid
    
        
    def propose_schedules(self, trades):
        return super().propose_schedules(trades)
    
    def predict_cost(self, vessel, trade):
        
        # calculate loading and unloading costs
        loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        loading_costs = vessel.get_loading_consumption(loading_time)
        unloading_costs = vessel.get_unloading_consumption(loading_time)
        
        # calculate travel costs
        travel_distance = self.headquarters.get_network_distance(
            trade.origin_port, trade.destination_port)
        travel_time = vessel.get_travel_time(travel_distance)
        travel_cost = vessel.get_laden_consumption(travel_time, vessel.speed)
        
        # calculate total cost
        total_cost = loading_costs + unloading_costs + travel_cost  
        
        return total_cost
    
    def log(
            self,
            text: str
    ):
        print(f"{self.name} # {text}")