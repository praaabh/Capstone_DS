from __future__ import annotations

from typing import List
import pandas as pd
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.allocation_algorithm.allocate_metaclass import AllocateMetaClass


class AllocateUniversal(metaclass=AllocateMetaClass):
    def __init__(self, data: pd.DataFrame, file_name: str, **kwargs):
        self._data = data
        self._file_name = file_name
        # Extracting City, Product and Bucket
        self.city = kwargs.get("city", None)
        self.product = kwargs.get("product", None)
        self.bucket = kwargs.get("bucket", None)

        self.pre_process(self.city, self.product, self.bucket)
        self.calculate_minimum_allocation()
        self.amounts = self.amounts.tolist()
        self.kpi = list(self.kpi)

    def __call__(
        self, percentages: List[float] | None = None, free_allocation: bool = True
    ):
        self._allocate(percentages, free_allocation)

    def _allocate(
        self, percentages: List[float] | None = None, free_allocation: bool = True
    ):
        # if no percentages are given
        if percentages is None:
            percentages = np.array(self.kpi) / np.sum(self.kpi)

        assert (
            len(percentages) == self.num_agencies
        ), "Number of percentages must be equal to number of agencies"

        # Defining the Model
        model = gp.Model("Allocation")
        X = model.addVars(
            self.num_accounts,
            self.num_agencies,
            vtype=GRB.BINARY,
            name="Allocation Decision variables",
        )
        # --------------------------------------------------------------------------------------------------
        # Constraint - 1:
        for idx, a in enumerate(self.agencies):
            MIN = min(
                max(
                    self._agency_config[a]["min"] * self.num_accounts,
                    0.90 * percentages[idx] * self.num_accounts,
                ),
                self.minimum_accounts[a],
            )
            MAX = max(
                self._agency_config[a]["max"] * self.num_accounts,
                1.1 * percentages[idx] * self.num_accounts,
            )

            model.addConstr(
                gp.quicksum(X[i, idx] for i in range(self.num_accounts)) >= MIN,
                name="Minimum Limit",
            )
            model.addConstr(
                gp.quicksum(X[i, idx] for i in range(self.num_accounts)) <= MAX,
                name="Maximum Limit",
            )

        # Constraint - 2:
        for i in range(self.num_accounts):
            model.addConstr(
                gp.quicksum(X[i, a] for a in range(self.num_agencies)) == 1,
                name="Should be allocated to one agency",
            )
            if not free_allocation:
                model.addConstr(
                    gp.quicksum(
                        X[i, a_idx]
                        for a_idx, a in enumerate(self.agencies)
                        if a in self.possible_agencies[i]
                    )
                    == 1,
                    name="Restricted Agencies Allocation",
                )

        # Constraint - 3:
        _avg_balance_accounts = sum(self.amounts) / self.num_accounts
        for a in range(self.num_agencies):
            _balance_allocated = gp.quicksum(
                X[i, a] * self.amounts[i] for i in range(self.num_accounts)
            )
            # Lower and upper limit calculations
            if free_allocation:
                _lower_limit = (
                    0.925
                    * _avg_balance_accounts
                    * gp.quicksum(X[i, a] for i in range(self.num_accounts))
                )
            else:
                _lower_limit = min(
                    0.925 * _avg_balance_accounts,
                    self.minimum_average_balance[self.agencies[a]],
                ) * gp.quicksum(X[i, a] for i in range(self.num_accounts))

            _upper_limit = (
                1.075
                * _avg_balance_accounts
                * gp.quicksum(X[i, a] for i in range(self.num_accounts))
            )

            model.addConstr(
                _balance_allocated >= _lower_limit,
                name="Balance Limit Lower Limit",
            )

            model.addConstr(
                _balance_allocated <= _upper_limit,
                name="Balance Limit Upper Limit",
            )
        # --------------------------------------------------------------------------------------------------
        # Objective
        resolved_amount = gp.quicksum(
            self.kpi[a] * X[i, a] * self.amounts[i]
            for i in range(self.num_accounts)
            for a in range(self.num_agencies)
        )
        # Set Objective
        model.setObjective(resolved_amount, sense=GRB.MAXIMIZE)
        model.Params.MIPGap = 0.05
        model.Params.LogToConsole = 1
        model.optimize()
        # --------------------------------------------------------------------------------------------------
        if model.status == GRB.OPTIMAL:
            # Post Processing
            _x = np.zeros((self.num_accounts, self.num_agencies))

            for i, acc in enumerate(self.accounts):
                for j, agn in enumerate(self.agencies):
                    _x[i, j] = 1 if X[i, j].X > self._tolerance else 0

            _ = self.post_process(
                _x, user_percentages=percentages, out_file_name=self._file_name
            )
        else:
            print(
                f"UNABLE TO SOLVE THE OPTIMIZATION PROBLEM, MODEL STATUS: {model.status} \n"
            )
