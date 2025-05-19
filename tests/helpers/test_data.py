import pandas as pd


def get_mock_claims_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_age": [45, 52, 30, 60, 33],
            "total_charge_amount": [100.0, 300.0, 450.0, 200.0, 120.0],
            "claim_type": [1, 0, 1, 1, 0],
            "claim_age_days": [10, 20, 5, 8, 15],
            "provider_type": [0, 1, 0, 1, 0],
            "billing_provider_specialty": [1, 2, 2, 1, 0],
            "facility_code": [0, 1, 0, 1, 1],
            "denied": [0, 1, 0, 1, 0],
        }
    )
