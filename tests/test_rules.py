import pandas as pd
from ClaimFlowEngine.denial_rules.rules import apply_denial_rules


def test_apply_denial_rules_sample():

    claim = pd.Series(
        {
            "patient_id": "PT001",
            "procedure_date": "2024-12-01",
            "cpt_code": "70450",  # CT scan (auth)
            "icd10_code": "R51",  # Headache (vague diagnosis)
            "billed_amount": 1200.00,  # Triggers Authorization
            "patient_age": 17,
            "days_since_procedure": 95,
        }
    )

    result = apply_denial_rules(claim)

    print("--- Denial Rules ---")
    print(f"Flag: {result['denial_flag']}")
    print(f"Reason: {result['denial_reason']}")
    print(f"Code: {result['denial_code']}")
    print(f"Severity: {result['denial_severity']}")
    print(f"Auto Appealable: {result['auto_appealable']}")
    print(f"Routing Tag: {result['routing_tag']}")


if __name__ == "__main__":
    test_apply_denial_rules_sample()
